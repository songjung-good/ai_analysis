import json
import os
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
        source_id=f"web:{i}" if url else f"tech:{i}",
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
        benchmark_context=[claim],
        limitations=[claim],
        evidence_level="high",
        missing_items=[],
    )


def grade(ids, sufficient, company=False, missing=()):
    return technical.RelevanceGrade(
        relevant_ids=list(ids),
        sufficient=sufficient,
        company_specific=company,
        missing=list(missing),
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
        llm = FakeLLM([grade(["S1", "S2"], True, company=True)], analysis_output(["S1"]))
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
            [grade(["S1"], False, missing=["성능 근거"]), grade([], False), grade(["S1"], True)],
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
            {e.source_id for e in result["references"]}, {"tech:1", "web:10"}
        )

    def test_general_docs_without_company_evidence_go_to_web(self):
        llm = FakeLLM(
            [grade(["S1", "S2"], True, company=False), grade(["S1"], True, company=True)],
            analysis_output(["S3"]),
        )
        tech, web = self.patch(
            llm, [doc(1), doc(2)], web_docs=[doc(10, url="https://example.com/a")]
        )

        result = technical.run(STATE)

        self.assertEqual(tech.call_count, 2)  # 문헌은 충분하므로 재작성하지 않음
        self.assertEqual(web.call_count, 2)
        retrieval = result["technical_analysis"]["retrieval"]
        self.assertEqual(retrieval["rewrite_count"], 0)
        self.assertTrue(retrieval["used_web_fallback"])
        # 웹 근거(기업 고유 자료)를 인용했으므로 LLM이 매긴 level을 유지
        self.assertEqual(result["technical_analysis"]["evidence_level"], "high")

    def test_literature_only_caps_evidence_level_to_low(self):
        llm = FakeLLM([grade(["S1", "S2"], True, company=True)], analysis_output(["S1"]))
        self.patch(llm, [doc(1), doc(2)])

        analysis = technical.run(STATE)["technical_analysis"]

        self.assertEqual(analysis["evidence_level"], "low")
        self.assertEqual(analysis["benchmark_context"][0]["source_ids"], ["tech:1"])

    def test_unverified_claims_are_dropped(self):
        llm = FakeLLM([grade(["S1", "S2"], True, company=True)], analysis_output(["S1"]))
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
        llm = FakeLLM([grade(["S1", "S2"], True, company=True)], analysis_output(["S1"]))
        self.patch(llm, [doc(1), doc(2)])

        result = guard_node("technical_analysis", technical.run)(STATE)

        self.assertEqual(set(result), {"technical_analysis", "references"})


# ---------------------------------------------------------------------------
# 실제 API 스모크 테스트 (평소에는 skip)
#
# 실행 조건: .env에 OPENAI_API_KEY, OPENAI_MODEL, TAVILY_API_KEY가 있고
#            scripts/ingest_tech_docs.py로 tech_docs를 적재한 상태
# 실행 방법: RUN_LIVE_TESTS=1 PYTHONPATH=src python -m unittest tests.test_technical -v
# ---------------------------------------------------------------------------

# 1·2번 Agent가 만들어 줄 입력을 임시로 직접 작성 (설계서 C절 카본식스 예시).
# master 병합 후 전체 그래프로 실행할 때는 이 테스트를 지우거나 skip 상태로 둔다.
LIVE_STATE = {
    "selected_startup": {
        "name": "카본식스",
        "product": "시그마키트 (센서 글러브 시연 기반 로봇 작업 학습 솔루션)",
        "funding_stage": "Series A",
        "team": ["문태연 대표 (전 수아랩 사업총괄 부대표)"],
    },
    "profile": {
        "subdomain": "제조 현장 로봇 자동화",
        "paying_customer": "제조 기업",
        "problem": "비정형 반복 작업의 자동화 어려움과 인력 부족",
    },
}


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "실제 API 호출 테스트: RUN_LIVE_TESTS=1 일 때만 실행",
)
class TechnicalLiveSmokeTest(unittest.TestCase):
    """실제 LLM·벡터DB·Tavily로 기술·제품 검증 Agent를 한 번 실행한다."""

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parents[1] / ".env")
        cls.result = guard_node("technical_analysis", technical.run)(LIVE_STATE)
        cls.analysis = cls.result["technical_analysis"]
        if os.getenv("LIVE_TESTS_VERBOSE", "1") == "1":
            _print_live_result(cls.result)

    def test_writes_only_owned_state_keys(self):
        self.assertEqual(set(self.result), {"technical_analysis", "references"})

    def test_output_has_required_fields(self):
        for key in (
            "summary",
            "core_technologies",
            "performance_evidence",
            "benchmark_context",
            "limitations",
            "evidence_level",
            "missing_items",
            "retrieval",
        ):
            self.assertIn(key, self.analysis)
        self.assertIn(
            self.analysis["evidence_level"], {"high", "medium", "low", "insufficient"}
        )

    def test_limitations_and_references_are_present(self):
        self.assertTrue(self.analysis["limitations"], "기술적 한계는 필수 항목")
        self.assertTrue(self.result["references"], "인용된 출처가 없음")

    def test_every_cited_id_is_in_references(self):
        ref_ids = {ev.source_id for ev in self.result["references"]}
        for key in ("core_technologies", "performance_evidence", "benchmark_context", "limitations"):
            for claim in self.analysis[key]:
                self.assertTrue(set(claim["source_ids"]) <= ref_ids, claim)


def _print_live_result(result):
    analysis = result["technical_analysis"]
    print("\n=== retrieval ===")
    print(json.dumps(analysis["retrieval"], ensure_ascii=False, indent=2))
    print("\n=== evidence_level ===", analysis["evidence_level"])
    print("\n=== summary ===\n", analysis["summary"])
    for key in ("core_technologies", "performance_evidence", "benchmark_context", "limitations"):
        print(f"\n=== {key} ({len(analysis[key])}) ===")
        for item in analysis[key]:
            print("-", item["statement"], item["source_ids"])
    print("\n=== missing_items ===", analysis["missing_items"])
    print(f"\n=== references ({len(result['references'])}) ===")
    for ev in result["references"]:
        print("-", ev.title, f"p.{ev.page}" if ev.page else ev.url)


if __name__ == "__main__":
    unittest.main()
