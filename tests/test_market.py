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
                if schema is market.CompetitorDetails:
                    return market.CompetitorDetails(items=[
                        market.CompetitorDetail(
                            name=line[4:], type="국내 스타트업", product=f"{line[4:]} 제품",
                            target_customer="제조", differentiator="차이",
                        )
                        for line in prompt.splitlines() if line.startswith("### ")
                    ])
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
            "customer_problem": "조립 공정 자동화",
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

    def test_curate_drops_duplicate_competitor_names(self):
        competitor = self._analysis().competitors[0]
        analysis = self._analysis().model_copy(update={"competitors": [
            competitor, competitor.model_copy(update={"name": "Tommoro Robotics "}),
        ]})
        docs = {"b": evidence("b", excerpt="Tommoro Robotics")}
        curated, _ = market.curate(analysis, docs, "PLAIF")
        self.assertEqual([c.name for c in curated.competitors], ["Tommoro Robotics"])

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

    def test_questions_skip_unknown_profile_fields(self):
        profile = {"subdomain": "물류 로봇", "paying_customer": "정보 부족", "customer_problem": "정보 부족"}
        questions = market.build_questions({"name": "A"}, profile)
        self.assertNotIn("정보 부족", " ".join(questions.values()))
        self.assertIn("주요 고객", questions["demand"])

    def test_rejects_missing_profile_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "profile.subdomain"):
            market.analyze({"selected_startup": {"name": "A"}}, llm=FakeLLM(set()))

    def test_low_evidence_level_when_topics_lack_evidence(self):
        llm = FakeLLM(set(), self._analysis().model_copy(update={"market_size": [], "growth": [], "demand_drivers": [], "competitors": []}))
        result = market.analyze(self.STATE, llm=llm, search=lambda q: [evidence("a")], web=lambda q: [])
        self.assertEqual(result["market_analysis"]["evidence_level"], "low")
        self.assertEqual(result["references"], [])
        self.assertTrue(all(r["used_web"] for r in result["market_analysis"]["retrieval"].values()))



MAP_EXCERPT = (
    "#### 제조·산업 (Manufacturing)\n"
    "- 제조·산업 로봇 스타트업 (Robotics): Deft Robotics, Diden Robotics(디든로보틱스), "
    "ROBROS, Holiday Robotics(홀리데이로보틱스), PLAIF(플라잎)\n"
    "- 제조·산업 AI·SW 플랫폼 스타트업 (AI & SW Platforms): RLWRLD(리얼월드), Config\n"
    "#### 물류·유통 (Logistics)\n"
    "- 물류·유통 자율주행 스타트업 (Autonomous Mobility): TWINNY, RideFlux"
)
NARRATIVE_EXCERPT = "홀리데이로보틱스는 1,500억 원을 유치했고 디든로보틱스는 승월로봇으로 70억 원을 유치했다."


def landscape_search(query):
    if "투자 유치 사례" in query:
        return [evidence("story", "2026 피지컬 AI 스타트업맵", 5, NARRATIVE_EXCERPT), evidence("map", "map", 8, MAP_EXCERPT)]
    return [evidence("map", "2026 피지컬 AI 스타트업맵", 8, MAP_EXCERPT), evidence("x")]


class LandscapeTests(unittest.TestCase):
    def test_segment_from_subdomain_or_explicit_profile(self):
        self.assertEqual(market.resolve_segment({"subdomain": "물류 로봇"}), ("물류·유통", ("로봇", "자율주행")))
        self.assertIsNone(market.resolve_segment({"subdomain": "휴머노이드"}))
        self.assertEqual(
            market.resolve_segment({"subdomain": "기타", "industry": "건설·인프라", "tech_type": "드론·UAM"}),
            ("건설·인프라", ("드론·UAM",)),
        )

    def test_landscape_takes_only_segment_rows(self):
        rows, map_evidence, narrative = market.fetch_landscape(("제조·산업", ("로봇",)), landscape_search)
        self.assertEqual([(r.industry, r.tech_type) for r in rows], [("제조·산업", "로봇")])
        self.assertEqual(rows[0].companies[:2], ["Deft Robotics", "Diden Robotics(디든로보틱스)"])
        self.assertEqual(set(map_evidence), {"map"})
        self.assertEqual(set(narrative), {"story"})  # map chunk is not narrative

    def test_selection_prefers_described_companies_and_is_repeatable(self):
        rows, _, narrative = market.fetch_landscape(("제조·산업", ("로봇",)), landscape_search)
        selected = market.select_competitors(rows, narrative, "플라잎(PLAIF)")
        self.assertEqual(
            selected,
            ["Diden Robotics(디든로보틱스)", "Holiday Robotics(홀리데이로보틱스)", "Deft Robotics", "ROBROS"],
        )
        self.assertEqual(selected, market.select_competitors(rows, narrative, "플라잎(PLAIF)"))

    def test_describes_only_companies_named_in_sentences(self):
        rows, map_evidence, narrative = market.fetch_landscape(("제조·산업", ("로봇",)), landscape_search)
        evidence = {**map_evidence, **narrative}
        self.assertEqual(
            market.company_snippets("Holiday Robotics(홀리데이로보틱스)", evidence),
            [("story", "홀리데이로보틱스는 1,500억 원을 유치했고 디든로보틱스는 승월로봇으로 70억 원을 유치했다.")],
        )
        self.assertEqual(market.company_snippets("ROBROS", evidence), [])  # map list alone is not a description

        llm = FakeLLM(set())
        competitors = market.describe_competitors(
            llm, {"name": "PLAIF"}, ["ROBROS", "Holiday Robotics(홀리데이로보틱스)"], rows, evidence
        )
        self.assertEqual([c.name for c in competitors], ["ROBROS", "Holiday Robotics(홀리데이로보틱스)"])
        self.assertEqual((competitors[0].product, competitors[0].source_ids), ("근거 없음", ["map"]))
        self.assertEqual(competitors[1].source_ids, ["map", "story"])
        self.assertNotEqual(competitors[1].product, "근거 없음")
        described_prompt = [p for name, p in llm.prompts if name == "CompetitorDetails"][0]
        self.assertNotIn("### ROBROS", described_prompt)

    def test_other_competitors_excludes_selected(self):
        base = MarketAgentTests()._analysis().competitors[0]
        verified = [base.model_copy(update={"name": "홀리데이로보틱스"}), base.model_copy(update={"name": "씨메스로보틱스"})]
        self.assertEqual(
            market.other_competitors(verified, ["Holiday Robotics(홀리데이로보틱스)"]), ["씨메스로보틱스"]
        )

    def test_analyze_outputs_landscape_for_mapped_subdomain(self):
        state = {
            "selected_startup": {"name": "플라잎(PLAIF)"},
            "profile": {"subdomain": "산업용·제조 로봇", "paying_customer": "정보 부족", "customer_problem": "정보 부족"},
        }
        analysis = MarketAgentTests()._analysis().model_copy(update={"competitors": [], "market_size": [], "growth": []})
        result = market.analyze(state, llm=FakeLLM({"map", "story", "x"}, analysis), search=landscape_search, web=lambda q: [])
        landscape = result["market_analysis"]["competitor_landscape"]
        self.assertEqual(landscape["segment"], "제조·산업 로봇")
        self.assertEqual(landscape["company_count"], 5)
        self.assertEqual(landscape["selected"][0], "Diden Robotics(디든로보틱스)")
        self.assertEqual([c["name"] for c in result["market_analysis"]["competitors"]], landscape["selected"])
        self.assertEqual(landscape["evidence_sentences"]["Deft Robotics"], [])
        self.assertIn("1,500억", landscape["evidence_sentences"]["Holiday Robotics(홀리데이로보틱스)"][0])
        self.assertIn("map", {e.source_id for e in result["references"]})



def web_evidence(source_id, url, excerpt="근거"):
    return Evidence(source_id=source_id, title=url, url=url, excerpt=excerpt)


class SourcePolicyTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(market.source_tier(evidence("pdf")), 1)
        self.assertEqual(market.source_tier(web_evidence("a", "https://www.kiet.re.kr/research/1")), 1)
        self.assertEqual(market.source_tier(web_evidence("b", "https://ifr.org/news")), 1)
        self.assertEqual(market.source_tier(web_evidence("c", "https://www.etnews.com/2025")), 2)
        self.assertEqual(market.source_tier(web_evidence("d", "https://www.gminsights.com/robot")), 3)

    def test_trusted_first_widens_only_when_short_and_drops_blogs(self):
        calls = []

        def search(query, domains=None, max_results=5):
            calls.append(domains is not None)
            if domains:
                return [web_evidence("t1", "https://www.etnews.com/1")]
            return [
                web_evidence("t1", "https://www.etnews.com/1"),
                web_evidence("blog", "https://someone.tistory.com/2"),
                web_evidence("mill", "https://www.gminsights.com/3"),
            ]

        results = market.trusted_first_search("협동로봇 시장", search)
        self.assertEqual(calls, [True, False])
        self.assertEqual([e.source_id for e in results], ["t1", "mill"])

        calls.clear()
        full = lambda q, domains=None, max_results=5: [web_evidence(f"t{i}", f"https://mk.co.kr/{i}") for i in range(3)]
        market.trusted_first_search("협동로봇 시장", lambda *a, **k: (calls.append(True), full(*a, **k))[1])
        self.assertEqual(len(calls), 1)

    def _figure(self, value, sid, year="2024"):
        return market.MarketFigure(metric="협동로봇 시장 규모", value=value, unit="억 달러", year=year, region="세계", source_ids=[sid])

    def test_figures_need_tier_1_or_2_backing(self):
        docs = {
            "news": web_evidence("news", "https://www.hankyung.com/1", "시장 규모 19억 달러"),
            "mill": web_evidence("mill", "https://www.researchnester.com/1", "시장 규모 59.28억 달러"),
        }
        kept, unverified, low, _ = market._verified_figures(
            [self._figure(19, "news"), self._figure(59.28, "mill"), self._figure(7, "news")], docs
        )
        self.assertEqual([f.value for f in kept], [19])
        self.assertEqual(low, ["협동로봇 시장 규모 59.28억 달러 (2024)"])
        self.assertEqual(unverified, ["협동로봇 시장 규모 7억 달러 (2024)"])

    def test_korean_compound_amounts_match_decimal_values(self):
        self.assertEqual(
            market.with_decimal_forms("2024년 20억 3천만 달러에서 132억 5천만 달러, 18억 5,800만 대, 6조 8,158억 원"),
            "2024년 20억 3천만 달러에서 132억 5천만 달러, 18억 5,800만 대, 6조 8,158억 원 6.8158조 원 20.3억 달러 132.5억 달러 18.58억 대",
        )
        docs = {"pdf": evidence("pdf", excerpt="휴머노이드 시장은 2029년 132억 5천만 달러로 성장")}
        kept, unverified, _, _ = market._verified_figures([self._figure(132.5, "pdf", "2029(전망)")], docs)
        self.assertEqual((len(kept), unverified), (1, []))

    def test_wrong_chunk_citation_is_repointed_only_with_value_and_unit(self):
        docs = {
            "chunk0": evidence("chunk0", excerpt="모건스탠리의 휴머노이드 100 분석"),
            "chunk1": evidence("chunk1", excerpt="골드만삭스는 2035년까지 380억 달러에 이를 것으로 전망"),
            "noise": evidence("noise", excerpt="2023년 설치 380대"),
        }
        kept, unverified, _, recited = market._verified_figures(
            [self._figure(380, "chunk0", "2035(전망)"), self._figure(38, "chunk0", "2030")], docs
        )
        self.assertEqual([f.source_ids for f in kept], [["chunk1"]])
        self.assertEqual(recited, ["협동로봇 시장 규모 380억 달러 (2035(전망))"])
        self.assertEqual(unverified, ["협동로봇 시장 규모 38억 달러 (2030)"])

        shipments = market.MarketFigure(metric="출하량", value=1.4e6, unit="대", year="2035", region="세계", source_ids=["chunk0"])
        docs["chunk2"] = evidence("chunk2", excerpt="로봇 출하량은 4배 증가하여 140만 대에 도달")
        kept, _, _, _ = market._verified_figures([shipments], docs)
        self.assertEqual([f.source_ids for f in kept], [["chunk2"]])

    def test_expanded_units_still_match_source(self):
        docs = {"ifr": evidence("ifr", excerpt="(1,000 units) 천 대: 2023년 541, 2024년 542")}
        figure = market.MarketFigure(metric="설치 대수", value=542000, unit="대", year="2024", region="세계", source_ids=["ifr"])
        kept, unverified, _, _ = market._verified_figures([figure, figure.model_copy(update={"value": 543000})], docs)
        self.assertEqual([f.value for f in kept], [542000])
        self.assertEqual(unverified, ["설치 대수 543000대 (2024)"])

        cobots = {"ifr": evidence("ifr", excerpt="협동로봇 2024년 64,542대")}
        scaled = figure.model_copy(update={"value": 64.542, "unit": "천 대"})
        kept, _, _, _ = market._verified_figures([scaled], cobots)
        self.assertEqual(len(kept), 1)

    def test_repoints_korean_compound_amount_with_unit(self):
        docs = {
            "chunk0": evidence("chunk0", excerpt="휴머노이드 생태계 분석"),
            "chunk1": evidence("chunk1", excerpt="2029년 132억 5천만 달러로 성장할 전망"),
        }
        kept, _, _, recited = market._verified_figures([self._figure(132.5, "chunk0", "2029(전망)")], docs)
        self.assertEqual([f.source_ids for f in kept], [["chunk1"]])
        self.assertEqual(len(recited), 1)

    def test_unitless_or_zero_figures_are_not_matched_loosely(self):
        docs = {"a": evidence("a", excerpt="4개 기업, 2024년 +0% 성장"), "b": evidence("b", excerpt="0.4배 증가")}
        figure = self._figure(4, "a")
        kept, unverified, _, _ = market._verified_figures(
            [figure.model_copy(update={"unit": ""}), figure.model_copy(update={"value": 0, "unit": "배", "source_ids": ["b"]}),
             figure.model_copy(update={"value": 0, "unit": "%"})],
            docs,
        )
        self.assertEqual([(f.value, f.unit) for f in kept], [(0, "%")])
        self.assertEqual(len(unverified), 2)

    def test_conflicting_values_keep_better_tier(self):
        docs = {
            "news": web_evidence("news", "https://www.mk.co.kr/1"),
            "pdf": evidence("pdf"),
        }
        kept, conflicts = market.resolve_conflicts(
            [self._figure(59.28, "news"), self._figure(19.9, "pdf"), self._figure(21, "pdf", year="2025")], docs
        )
        self.assertEqual([(f.value, f.year) for f in kept], [(19.9, "2024"), (21, "2025")])
        self.assertEqual(conflicts, ["협동로봇 시장 규모 59.28억 달러 (2024) vs 19.9억 달러"])


if __name__ == "__main__":
    unittest.main()
