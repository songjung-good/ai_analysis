import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents import market
from ai_investment.contracts import guard_node
from ai_investment.crag import (
    RelevanceGrade,
    RewrittenQuery,
    build_crag,
    run_crag,
)
from ai_investment.models import Evidence


def evidence(source_id, title="IFR", page=8, excerpt=None):
    return Evidence(
        source_id=source_id, title=title, page=page, excerpt=excerpt or f"근거 {source_id}"
    )


class FakeLLM:
    """Returns canned structured outputs keyed by schema."""

    def __init__(self, relevant_ids, analysis=None):
        self.relevant_ids = set(relevant_ids)
        self.analysis = analysis
        self.prompts = []

    def with_structured_output(self, schema):
        llm = self

        class Runnable:
            def invoke(self, prompt):
                llm.prompts.append((schema.__name__, prompt))
                if schema is RelevanceGrade:
                    return RelevanceGrade(
                        relevant_ids=[sid for sid in llm.relevant_ids if f"[{sid}]" in prompt]
                    )
                if schema is RewrittenQuery:
                    return RewrittenQuery(query="rewritten query")
                return llm.analysis

        return Runnable()


class CragTests(unittest.TestCase):
    def _graph(self, llm, vector, web):
        from ai_investment.crag import llm_grader, llm_rewriter

        return build_crag(
            retrieve=vector, grade=llm_grader(llm), rewrite=llm_rewriter(llm), web_search=web
        )

    def test_sufficient_vector_result_skips_rewrite_and_web(self):
        llm = FakeLLM({"a", "b"})
        web_calls = []
        graph = self._graph(llm, lambda q: [evidence("a"), evidence("b"), evidence("c")], web_calls.append)
        result = run_crag(graph, "question")
        self.assertEqual([e.source_id for e in result.evidence], ["a", "b"])
        self.assertEqual(result.queries, ["question"])
        self.assertFalse(result.used_web)
        self.assertEqual(web_calls, [])

    def test_rewrites_once_then_falls_back_to_web(self):
        llm = FakeLLM({"w1", "w2"})
        vector_queries = []

        def vector(query):
            vector_queries.append(query)
            return [evidence("x")]

        graph = self._graph(llm, vector, lambda q: [evidence("w1", "web", None), evidence("w2", "web", None)])
        result = run_crag(graph, "question")
        self.assertEqual(vector_queries, ["question", "rewritten query"])
        self.assertEqual(result.queries, ["question", "rewritten query", "web:rewritten query"])
        self.assertTrue(result.used_web)
        self.assertTrue(result.sufficient)

    def test_grader_ignores_ids_not_in_candidates(self):
        from ai_investment.crag import llm_grader

        class Liar(FakeLLM):
            def with_structured_output(self, schema):
                class Runnable:
                    def invoke(self, prompt):
                        return RelevanceGrade(relevant_ids=["a", "made-up"])

                return Runnable()

        self.assertEqual(llm_grader(Liar(set()))("q", [evidence("a")]), {"a"})


class MarketAgentTests(unittest.TestCase):
    STATE = {
        "selected_startup": {"name": "PLAIF", "product": "AI 양팔로봇"},
        "profile": {
            "subdomain": "제조 협동로봇",
            "paying_customer": "대기업 제조라인",
            "problem": "조립 공정 자동화",
        },
    }

    def _analysis(self):
        return market.MarketAnalysis(
            summary="협동로봇 설치가 늘고 있다.",
            market_size=[
                market.MarketFigure(metric="협동로봇 연간 설치", value=64542, unit="대", year="2024", region="세계", source_ids=["a"]),
                market.MarketFigure(metric="환각 수치", value=1, unit="조 원", year="2024", region="한국", source_ids=["nope"]),
            ],
            growth=[
                market.MarketFigure(metric="협동로봇 설치 증가율", value=12, unit="%", year="2024", region="세계", source_ids=["a", "nope"])
            ],
            demand_drivers=[market.Claim(claim="인력 부족", source_ids=["b"])],
            competitors=[
                market.Competitor(
                    name="Tommoro Robotics", type="국내 스타트업", product="AI 플랫폼",
                    target_customer="제조", differentiator="SW 중심", source_ids=["b"],
                )
            ],
            differentiation=market.Differentiation(strengths=[], weaknesses=[], entry_barriers=[]),
            missing_info=[],
        )

    def test_returns_only_owned_keys_and_cited_references(self):
        llm = FakeLLM({"a", "b", "unused"}, self._analysis())
        search = lambda q: [
            evidence("a", excerpt="협동로봇 64,542대, 전년 대비 +12%"),
            evidence("b", excerpt="제조·산업 AI·SW 플랫폼 스타트업: Tommoro Robotics, PLAIF(플라잎)"),
            evidence("unused"),
        ]
        node = guard_node("market_analysis", lambda s: market.analyze(s, llm=llm, search=search, web=lambda q: []))
        result = node(self.STATE)

        analysis = result["market_analysis"]
        self.assertEqual([f["metric"] for f in analysis["market_size"]], ["협동로봇 연간 설치"])
        self.assertEqual(analysis["growth"][0]["source_ids"], ["a"])
        self.assertEqual(analysis["validation"]["dropped_uncited_items"], 1)
        self.assertEqual([c["name"] for c in analysis["competitors"]], ["Tommoro Robotics"])
        self.assertEqual(analysis["evidence_level"], "high")
        self.assertEqual(analysis["subdomain"], "제조 협동로봇")
        self.assertEqual({e.source_id for e in result["references"]}, {"a", "b"})
        self.assertEqual(market.MarketAnalysisOutput.model_validate(analysis).model_dump(), analysis)

    def test_curate_drops_unverified_and_self_competitors(self):
        base = self._analysis()
        competitor = base.competitors[0]
        analysis = base.model_copy(update={"competitors": [
            competitor,
            competitor.model_copy(update={"name": "DidRen Robotics"}),
            competitor.model_copy(update={"name": "플라잎"}),
        ]})
        docs = {
            "b": evidence("b", excerpt="휴머노이드 시장 동향"),
            "map": evidence("map", excerpt="Tommoro Robotics, Diden Robotics(디든로보틱스)"),
        }
        curated, checks = market.curate(analysis, docs, "PLAIF(플라잎)")
        self.assertEqual([c.name for c in curated.competitors], ["Tommoro Robotics"])
        self.assertEqual(curated.competitors[0].source_ids, ["map"])
        self.assertEqual(checks["recited_competitors"], ["Tommoro Robotics"])
        self.assertEqual(checks["unverified_competitors"], ["DidRen Robotics"])
        self.assertTrue(checks["excluded_self_as_competitor"])

    def test_curate_drops_competitor_mapped_to_other_industry(self):
        competitor = self._analysis().competitors[0]
        analysis = self._analysis().model_copy(update={"competitors": [
            competitor.model_copy(update={"name": "Navifra"}),
            competitor.model_copy(update={"name": "Tommoro Robotics"}),
            competitor.model_copy(update={"name": "Figure AI"}),
        ]})
        docs = {
            "b": evidence("b", excerpt=(
                "- 물류·유통 AI·SW 플랫폼 스타트업 (AI & SW Platforms): Navifra, BISCAT\n"
                "- 제조·산업 AI·SW 플랫폼 스타트업 (AI & SW Platforms): Tommoro Robotics\n"
                "Figure AI는 시리즈C에서 10억 달러를 유치했다."
            )),
        }
        curated, checks = market.curate(analysis, docs, "PLAIF", industry="제조·산업")
        self.assertEqual([c.name for c in curated.competitors], ["Tommoro Robotics", "Figure AI"])
        self.assertEqual(checks["industry_mismatched_competitors"], ["Navifra"])

    def test_curate_keeps_latest_actual_figure_per_metric(self):
        figure = self._analysis().market_size[0]
        series = [
            figure.model_copy(update={"year": year, "value": value})
            for year, value in [("2022", 57966), ("2024", 64542), ("2028(전망)", 90000), ("2023", 57148)]
        ]
        docs = {"a": evidence("a", excerpt="2022년 57,966대, 2023년 57,148대, 2024년 64,542대, 2028년 90000대")}
        curated, _ = market.curate(
            self._analysis().model_copy(update={"market_size": series, "competitors": [], "growth": []}), docs, "PLAIF"
        )
        self.assertEqual([(f.year, f.value) for f in curated.market_size], [("2024", 64542)])

    def test_curate_drops_figure_absent_from_cited_evidence(self):
        docs = {"a": evidence("a", excerpt="협동로봇 64,542대, 전년 대비 +12%")}
        analysis = self._analysis().model_copy(update={"competitors": []})
        analysis = analysis.model_copy(update={"market_size": [
            analysis.market_size[0],
            analysis.market_size[0].model_copy(update={"metric": "시장 금액", "value": 3.5, "unit": "조 원"}),
        ]})
        curated, checks = market.curate(analysis, docs, "PLAIF")
        self.assertEqual([f.metric for f in curated.market_size], ["협동로봇 연간 설치"])
        self.assertEqual(checks["unverified_figures"], ["시장 금액 3.5조 원 (2024)"])

    def test_questions_use_profile(self):
        questions = market.build_questions(self.STATE["selected_startup"], self.STATE["profile"])
        self.assertEqual(set(questions), {"market", "demand", "competition"})
        self.assertIn("제조 협동로봇", questions["market"])
        self.assertIn("대기업 제조라인", questions["demand"])
        self.assertIn("조립 공정 자동화", questions["competition"])

    def test_rejects_missing_profile_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "profile.subdomain"):
            market.analyze({"selected_startup": {"name": "A"}}, llm=FakeLLM(set()))

    def test_low_evidence_level_when_topics_lack_evidence(self):
        llm = FakeLLM(set(), self._analysis().model_copy(update={"market_size": [], "growth": [], "demand_drivers": [], "competitors": []}))
        result = market.analyze(self.STATE, llm=llm, search=lambda q: [evidence("a")], web=lambda q: [])
        self.assertEqual(result["market_analysis"]["evidence_level"], "low")
        self.assertEqual(result["references"], [])
        self.assertTrue(all(r["used_web"] for r in result["market_analysis"]["retrieval"].values()))


if __name__ == "__main__":
    unittest.main()
