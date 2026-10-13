"""Supervisor 그래프의 동적 라우팅·평가 루프·종료 보장 테스트 (LLM 호출 없음).

실제 supervisor와 그래프 조립을 쓰고, 분석·보고서·평가 노드는 가짜로 대체한다.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.graph import AgentNodes, build_graph  # noqa: E402
from ai_investment.orchestration.supervisor import supervisor  # noqa: E402
from ai_investment.state import VIEWS, create_initial_state  # noqa: E402
from ai_investment.utils import logger  # noqa: E402


def _result(view, complete=True):
    return {
        "view": view, "summary": "stub", "claims": [],
        "missing_items": [] if complete else ["시장 규모"],
        "source_ids": ["s1", "s2"] if complete else ["s1"],
        "status": "done",
    }


def _verdict(passed, action, **extra):
    return {"passed": passed, "items": {}, "action": action,
            "target_views": [], "feedback": [], **extra}


class SupervisorGraphTests(unittest.TestCase):
    def setUp(self):
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.skipTest("langgraph is not installed")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(logger, "LOG_DIR", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

    def _run(self, evaluator, *, max_steps=20):
        calls = self.calls

        def view_node(view):
            def node(state):
                calls.append(view)
                reworked = bool(state.get("rework_hints", {}).get(view))
                incomplete = view == "market" and not reworked  # market만 첫 실행에서 근거 부족
                return {
                    "analyses": {view: _result(view, complete=not incomplete)},
                    "node_status": {view: "done"},
                    "node_errors": {view: None},
                    "references": [],
                }
            return node

        def discovery(state):
            startup = {"name": "A"}
            return {"candidates": [startup], "current_idx": 0,
                    "selected_startup": startup, "references": []}

        def profile(state):
            return {"profile": {}, "references": []}

        def decision(state):
            calls.append("investment_decision")
            return {"scores": {}, "investment_score": None, "decision": "hold",
                    "decision_reason": "stub",
                    "evaluations": [{"startup_name": "A", "score": None,
                                     "decision": "hold", "reason": "stub"}]}

        def report(state):
            calls.append("report_generation")
            verdict = state.get("eval_result") or {}
            limited = bool(verdict) and not verdict.get("passed", False)
            return {"report": "stub.pdf", "report_draft": {"limit_report": limited}}

        def counted_evaluator(state):
            calls.append("evaluator")
            return evaluator(state)

        nodes = AgentNodes(
            startup_discovery=discovery, customer_profile=profile,
            technical_analysis=view_node("technical"),
            business_analysis=lambda state: {},
            market_analysis=view_node("market"),
            investment_decision=decision, report_generation=report,
            supervisor=supervisor,
            stakeholder=view_node("stakeholder"), domain=view_node("domain"),
            evaluator=counted_evaluator,
        )
        state = create_initial_state(
            domain="Physical AI", criteria={}, max_iterations=1,
            max_steps=max_steps, trace_id="test-graph",
        )
        return build_graph(nodes).invoke(state, config={"recursion_limit": 100})

    @staticmethod
    def _always_pass(state):
        return {"eval_result": _verdict(True, "pass")}

    def test_only_insufficient_view_is_reworked(self):
        final = self._run(self._always_pass)
        self.assertEqual(set(self.calls[:4]), set(VIEWS))  # 첫 라운드: 4관점이 함께 실행
        self.assertEqual(self.calls.count("market"), 2)    # 근거 부족한 관점만 재실행
        for view in ("technical", "stakeholder", "domain"):
            self.assertEqual(self.calls.count(view), 1)
        self.assertEqual(final["rework_count"]["market"], 1)
        self.assertTrue(final["eval_result"]["passed"])

    def test_evaluator_rewrite_then_research_resumes_at_report(self):
        def evaluator(state):
            attempts = state.get("report_attempts", 0)
            if attempts == 0:  # 표현 문제 -> 보고서 재작성
                return {"eval_result": _verdict(False, "rewrite", feedback=["인용 보강"]),
                        "report_attempts": 1}
            if attempts == 1:  # 근거 문제 -> domain 재조사 후 보고서로 복귀
                return {"eval_result": _verdict(
                            False, "research", target_views=["domain"],
                            research_hints={"domain": ["운영 제약"]}),
                        "report_attempts": 2,
                        "resume_after_research": "report_generation"}
            return {"eval_result": _verdict(True, "pass")}

        final = self._run(evaluator)
        self.assertTrue(final["eval_result"]["passed"])
        self.assertEqual(final["report_attempts"], 2)
        self.assertEqual(final["rework_count"]["domain"], 1)
        self.assertEqual(self.calls.count("domain"), 2)
        self.assertEqual(self.calls.count("investment_decision"), 1)  # 재조사 후 투자 판단 반복 없음
        self.assertEqual(self.calls.count("report_generation"), 3)
        self.assertEqual(self.calls.count("evaluator"), 3)

    def test_always_failing_evaluator_ends_with_one_limit_report(self):
        def failing(state):
            return {"eval_result": _verdict(False, "rewrite", feedback=["항상 미달"]),
                    "report_attempts": state.get("report_attempts", 0) + 1}

        final = self._run(failing)
        self.assertFalse(final["eval_result"]["passed"])
        self.assertEqual(final["report_attempts"], 3)
        self.assertEqual(self.calls.count("evaluator"), 3)
        self.assertEqual(self.calls.count("report_generation"), 4)  # 초안 + 보완 2회 + 한계 보고서
        self.assertTrue(final["report_draft"]["limit_report"])
        self.assertEqual(self.calls[-1], "report_generation")       # 한계 보고서 후 평가 없이 종료

    def test_supervisor_step_cap_stops_rework(self):
        final = self._run(self._always_pass, max_steps=1)
        self.assertEqual(self.calls.count("market"), 1)  # 호출 상한 도달 -> 재작업 없이 진행
        self.assertEqual(final["rework_count"]["market"], 0)
        self.assertTrue(final["eval_result"]["passed"])


if __name__ == "__main__":
    unittest.main()
