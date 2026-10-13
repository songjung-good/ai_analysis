import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langgraph.graph import END, START, StateGraph

from ai_investment.contracts import guard_node
from ai_investment.control import select_next_candidate
from ai_investment.models import Evidence
from ai_investment.state import (
    GraphState, candidate_reset_update, create_initial_state, dedupe_by_source_id,
)


def view_result(view, summary="분석 결과"):
    return {
        "view": view, "summary": summary, "claims": [],
        "missing_items": [], "source_ids": [], "status": "done",
    }


class StateContractTests(unittest.TestCase):
    def test_reference_deduplication_preserves_excerpts_and_inputs(self):
        previous = [Evidence(source_id="same", title="자료", excerpt="첫 근거")]
        update = [
            Evidence(source_id="same", title="자료", excerpt="추가 근거"),
            Evidence(source_id="new", title="새 자료", excerpt="새 근거"),
        ]
        before = deepcopy((previous, update))
        merged = dedupe_by_source_id(previous, update)
        self.assertEqual([item.source_id for item in merged], ["same", "new"])
        self.assertIn("첫 근거", merged[0].excerpt)
        self.assertIn("추가 근거", merged[0].excerpt)
        self.assertEqual(dedupe_by_source_id(merged, update), merged)
        self.assertEqual((previous, update), before)

    def test_parallel_results_and_errors_merge_then_rework_replaces_one_view(self):
        builder = StateGraph(GraphState)
        evidence = Evidence(source_id="shared", title="공통 자료", excerpt="근거")
        for view in ("technical", "market"):
            def worker(_state, view=view):
                return {
                    "analyses": {view: {**view_result(view), "status": "failed"}},
                    "node_status": {view: "failed"},
                    "node_errors": {view: f"{view} 실패"},
                    "references": [evidence],
                }

            builder.add_node(view, guard_node(view, worker))
            builder.add_edge(START, view)

        def rework(state):
            self.assertEqual(set(state["analyses"]), {"technical", "market"})
            self.assertEqual(state["node_errors"], {
                "technical": "technical 실패", "market": "market 실패",
            })
            return {
                "analyses": {"technical": view_result("technical", "보완 결과")},
                "node_status": {"technical": "done"},
                "node_errors": {"technical": None},
            }

        builder.add_node("rework", guard_node("technical", rework))
        builder.add_edge(["technical", "market"], "rework")
        builder.add_edge("rework", END)
        result = builder.compile().invoke(create_initial_state(
            domain="Physical AI", criteria={}, max_iterations=1,
        ))
        self.assertEqual(result["analyses"]["technical"]["summary"], "보완 결과")
        self.assertEqual(result["analyses"]["market"]["summary"], "분석 결과")
        self.assertEqual(result["node_status"], {"technical": "done", "market": "failed"})
        self.assertEqual(result["node_errors"], {"technical": None, "market": "market 실패"})
        self.assertEqual(result["references"], [evidence])

    def test_candidate_reset_clears_reducer_fields_but_keeps_run_history(self):
        initial = create_initial_state(
            domain="Physical AI", criteria={}, max_iterations=2, trace_id="same-run",
        )
        evidence = Evidence(source_id="old", title="이전 자료", excerpt="이전 근거")
        evaluation = {"startup_name": "A", "score": None, "decision": "hold", "reason": "부족"}
        initial.update({
            "candidates": [{"name": "A"}, {"name": "B"}], "current_idx": 0,
            "selected_startup": {"name": "A"},
            "analyses": {"domain": view_result("domain", "A 분석")},
            "node_status": {"domain": "failed"}, "node_errors": {"domain": "A 실패"},
            "next_views": ["domain"], "rework_count": {"domain": 2},
            "rework_hints": {"domain": ["A 추가 조사"]}, "step_count": 7,
            "report_draft": {"summary": "A 보고서"}, "report": "A.pdf",
            "report_attempts": 2, "references": [evidence], "evaluations": [evaluation],
            "profile": {"customer": "A 고객"},
            "business_analysis": {"summary": "A 사업성"},
            "technical_analysis": {"summary": "A 기술"},
            "market_analysis": {"summary": "A 시장"},
            "scores": {"market": 4.0}, "investment_score": 4.0,
            "decision": "invest", "decision_reason": "A 결정",
            "resume_after_research": "report_generation",
            "eval_result": {
                "passed": False, "items": {}, "action": "research",
                "target_views": ["domain"], "feedback": [],
            },
        })
        builder = StateGraph(GraphState)
        builder.add_node("next_candidate", guard_node("next_candidate", lambda state: {
            **select_next_candidate(state), **candidate_reset_update(),
        }))
        builder.add_edge(START, "next_candidate")
        builder.add_edge("next_candidate", END)
        result = builder.compile().invoke(initial)
        self.assertEqual(result["selected_startup"], {"name": "B"})
        self.assertEqual(result["analyses"], {})
        self.assertEqual(result["node_status"], {})
        self.assertEqual(result["node_errors"], {})
        self.assertIsNone(result["eval_result"])
        self.assertIsNone(result["report_draft"])
        self.assertEqual(result["next_views"], [])
        self.assertEqual(result["rework_count"]["domain"], 0)
        self.assertEqual(result["rework_hints"]["domain"], [])
        self.assertEqual(result["report_attempts"], 0)
        self.assertEqual(result["report"], "")
        for field in ("profile", "technical_analysis", "business_analysis", "market_analysis", "scores"):
            self.assertEqual(result[field], {})
        self.assertIsNone(result["investment_score"])
        self.assertIsNone(result["decision"])
        self.assertEqual(result["decision_reason"], "")
        self.assertEqual(result["resume_after_research"], "investment_decision")
        self.assertEqual(result["step_count"], 7)
        self.assertEqual(result["trace_id"], "same-run")
        self.assertEqual(result["references"], [evidence])
        self.assertEqual(result["evaluations"], [evaluation])

    def test_new_workers_cannot_change_routing_and_evaluator_cannot_change_analysis(self):
        for node, update in (
            ("stakeholder", {"next_views": ["domain"]}),
            ("domain", {"rework_count": {"domain": 0}}),
            ("evaluator", {"analyses": {}}),
        ):
            with self.subTest(node=node), self.assertRaises(ValueError):
                guard_node(node, lambda _state: update)({})

    def test_workers_cannot_overwrite_other_view_results_status_or_errors(self):
        for update in (
            {"analyses": {"domain": view_result("domain")}},
            {"node_status": {"domain": "done"}},
            {"node_errors": {"domain": None}},
            {"analyses": {"stakeholder": view_result("domain")}},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                guard_node("stakeholder", lambda _state: update)({})


if __name__ == "__main__":
    unittest.main()
