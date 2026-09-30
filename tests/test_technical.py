import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents import technical  # noqa: E402
from ai_investment.contracts import guard_node  # noqa: E402
from ai_investment.models import Evidence  # noqa: E402


def doc(i, url=None):
    return Evidence(
        source_id=f"tech:{i}",
        title=f"Doc {i}",
        url=url,
        page=None if url else i,
        excerpt=f"evidence text {i}",
    )


STATE = {
    "selected_startup": {"name": "로봇스타트업", "product": "VLA 기반 매니퓰레이터"},
    "profile": {"subdomain": "제조 로봇"},
}


class FakeLLM:
    """schema별로 미리 정한 응답을 순서대로 돌려준다."""

    def __init__(self, grades, analysis=None):
        self.grades = list(grades)
        self.analysis = analysis
        self.calls = []

    def __call__(self, schema, prompt):
        self.calls.append(schema.__name__)
        if schema is technical.QueryPlan:
            return technical.QueryPlan(queries=["VLA 일반화 한계", "로봇 파운데이션 모델 성능"])
        if schema is technical.RelevanceGrade:
            return self.grades.pop(0)
        if schema is technical.TechnicalAnalysisOutput:
            return self.analysis
        raise AssertionError(schema)


def analysis_output(source_ids):
    claim = technical.Claim(statement="한계", basis="paper", source_ids=source_ids)
    return technical.TechnicalAnalysisOutput(
        summary="요약",
        core_technologies=[
            technical.TechClaim(
                statement="VLA 정책", layer="action", basis="paper", source_ids=source_ids
            ),
            technical.TechClaim(
                statement="자체 주장", layer="decision", basis="company_claim", source_ids=[]
            ),
        ],
        differentiation=[],
        performance_evidence=[
            technical.Claim(statement="근거 없는 주장", basis="third_party", source_ids=["S99"])
        ],
        limitations=[claim],
        evidence_level="medium",
        missing_items=[],
    )


def grade(ids, sufficient, missing=()):
    return technical.RelevanceGrade(
        relevant_ids=list(ids), sufficient=sufficient, missing=list(missing)
    )


class TechnicalAgentTests(unittest.TestCase):
    def patch(self, llm, tech_docs, web_docs=()):
        tech = mock.Mock(return_value=list(tech_docs))
        tech.__name__ = "search_tech_docs"
        web = mock.Mock(return_value=list(web_docs))
        web.__name__ = "web_search"
        self.enterContext(mock.patch.object(technical, "_structured", llm))
        self.enterContext(mock.patch.object(technical, "TOOLS", (tech, web)))
        return tech, web

    def test_sufficient_docs_skip_rewrite_and_web(self):
        llm = FakeLLM([grade(["S1", "S2"], True)], analysis_output(["S1"]))
        tech, web = self.patch(llm, [doc(1), doc(2), doc(3)])

        result = technical.run(STATE)

        web.assert_not_called()
        retrieval = result["technical_analysis"]["retrieval"]
        self.assertEqual(retrieval["rewrite_count"], 0)
        self.assertFalse(retrieval["used_web_fallback"])
        # 실제 인용한 S1만 references에 들어간다.
        self.assertEqual([e.source_id for e in result["references"]], ["tech:1"])

    def test_insufficient_docs_rewrite_then_web_fallback(self):
        llm = FakeLLM(
            [grade(["S1"], False, ["성능 근거"]), grade([], False), grade(["S1"], True)],
            analysis_output(["S1", "S2"]),
        )
        tech, web = self.patch(
            llm, [doc(1)], web_docs=[doc(10, url="https://example.com/a")]
        )

        result = technical.run(STATE)

        self.assertEqual(tech.call_count, 4)  # 질의 2개 x (최초 + 재작성 1회)
        self.assertEqual(web.call_count, 2)
        retrieval = result["technical_analysis"]["retrieval"]
        self.assertEqual(retrieval["rewrite_count"], 1)
        self.assertTrue(retrieval["used_web_fallback"])
        self.assertEqual(
            {e.source_id for e in result["references"]}, {"tech:1", "tech:10"}
        )

    def test_unverified_claims_are_dropped(self):
        llm = FakeLLM([grade(["S1", "S2"], True)], analysis_output(["S1"]))
        self.patch(llm, [doc(1), doc(2)])

        analysis = technical.run(STATE)["technical_analysis"]

        self.assertEqual(analysis["performance_evidence"], [])  # S99는 존재하지 않는 출처
        self.assertEqual(len(analysis["core_technologies"]), 2)  # company_claim은 유지
        self.assertEqual(analysis["limitations"][0]["source_ids"], ["tech:1"])

    def test_empty_vector_db_raises(self):
        self.patch(FakeLLM([]), [])
        with self.assertRaisesRegex(RuntimeError, "tech_docs"):
            technical.run(STATE)

    def test_no_evidence_skips_llm_analysis(self):
        llm = FakeLLM([grade([], False), grade([], False)])
        self.patch(llm, [doc(1)], web_docs=[])

        result = technical.run(STATE)

        self.assertNotIn("TechnicalAnalysisOutput", llm.calls)
        self.assertEqual(result["technical_analysis"]["evidence_level"], "insufficient")
        self.assertEqual(result["references"], [])

    def test_missing_startup_name_raises(self):
        with self.assertRaisesRegex(ValueError, "selected_startup"):
            technical.run({"selected_startup": {}})

    def test_writes_only_owned_state_keys(self):
        llm = FakeLLM([grade(["S1", "S2"], True)], analysis_output(["S1"]))
        self.patch(llm, [doc(1), doc(2)])

        result = guard_node("technical_analysis", technical.run)(STATE)

        self.assertEqual(set(result), {"technical_analysis", "references"})


if __name__ == "__main__":
    unittest.main()
