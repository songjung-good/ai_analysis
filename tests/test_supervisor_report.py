"""V2 report contract and evaluator-loop tests with no network calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pymupdf

from ai_investment.agents.report import ReportAgent, ReportDraft, _report_context, _write_report_draft
from ai_investment.contracts import NODE_WRITES, guard_node
from ai_investment.models import Evidence
from ai_investment.state import GraphState, VIEWS, create_initial_state
from test_report import sample_draft, sample_state


class SupervisorReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.agent = ReportAgent(output_dir=self.directory, draft_writer=lambda _: sample_draft())
        self.state = sample_state()

    def create_draft(self):
        self.state.update(guard_node("report_generation", self.agent.run)(self.state))

    def test_exact_shared_write_contract_and_serializable_delta(self):
        before = copy.deepcopy(self.state)
        result = self.agent.run(self.state)
        self.assertEqual(set(result), NODE_WRITES["report_generation"])
        self.assertTrue(set(result).issubset(GraphState.__annotations__))
        json.dumps(result)
        self.assertEqual(self.state, before)
        self.assertEqual(self.state["report_attempts"], 0)

    def test_v2_analyses_and_code_decision_reach_model_context(self):
        context = _report_context(self.state)
        self.assertEqual(context["analyses"], self.state["analyses"])
        self.assertEqual(context["investment_score"], 3.05)
        self.assertEqual(context["decision"], "conditional")
        self.assertEqual(context["scores"], self.state["scores"])
        self.assertEqual([s["source_id"] for s in context["references"]], ["used"])

    def test_v2_analyses_are_authoritative_over_stale_legacy_results(self):
        self.state["technical_analysis"] = {"summary": "previous candidate"}
        context = _report_context(self.state)
        self.assertNotIn("technical_analysis", context)
        self.assertEqual(context["analyses"], self.state["analyses"])

    def test_decision_basis_replaces_recommendation(self):
        self.assertIn("decision_basis", ReportDraft.model_fields)
        self.assertNotIn("recommendation", ReportDraft.model_fields)
        draft = ReportDraft.model_validate({"decision_basis": [{"text": "점수 근거", "source_ids": ["used"]}]})
        self.assertTrue(draft.decision_basis)

    def test_missing_view_rejected(self):
        draft = sample_draft()
        draft.stakeholder = []
        self.agent.draft_writer = lambda _: draft
        with self.assertRaisesRegex(ValueError, "stakeholder"):
            self.agent.run(self.state)

    def test_partial_source_id_is_not_matched(self):
        draft = sample_draft()
        draft.summary[0].source_ids = ["use"]
        self.agent.draft_writer = lambda _: draft
        with self.assertRaisesRegex(ValueError, "unknown source_ids"):
            self.agent.run(self.state)
        self.assertEqual(draft.summary[0].source_ids, ["use"])

    def test_prior_candidate_sources_are_not_exposed_or_citable(self):
        draft = sample_draft()
        draft.summary[0].source_ids = ["unused"]
        self.agent.draft_writer = lambda _: draft
        with self.assertRaisesRegex(ValueError, "outside current candidate"):
            self.agent.run(self.state)

    def test_no_current_sources_never_falls_back_to_previous_candidate_pool(self):
        self.state["analyses"] = {}
        self.state["profile"] = {}
        self.assertEqual(_report_context(self.state)["references"], [])

    def test_input_and_injected_writer_objects_not_mutated(self):
        before = copy.deepcopy(self.state)
        draft = sample_draft()
        original = draft.model_dump()
        def writer(state):
            state["analyses"].clear()
            return draft
        self.agent.draft_writer = writer
        self.agent.run(self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(draft.model_dump(), original)

    def test_final_failed_verdict_renders_limits_without_model_or_pass_gate(self):
        self.create_draft()
        self.state["report_attempts"] = 3
        self.state["eval_result"] = {"passed": False, "items": {"groundedness": {"passed": False, "reason": "시장 근거 부족"}},
                                     "action": "research", "target_views": ["market"], "feedback": ["출처 다양성 보완"]}
        self.agent.draft_writer = lambda _: self.fail("finalize must not invoke LLM")
        before = copy.deepcopy(self.state)
        result = guard_node("report_generation", self.agent.finalize)(self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(self.state["report_attempts"], 3)
        for key in ("summary", *VIEWS, "risks", "decision_basis"):
            self.assertEqual(result["report_draft"][key], self.state["report_draft"][key])
        with pymupdf.open(result["report"]) as pdf:
            text = "\n".join(p.get_text() for p in pdf)
            for value in ("FINAL / NOT PASSED", "품질 평가 미통과", "시장 근거 부족", "근거 부족 항목", self.state["trace_id"]):
                self.assertIn(value, text)

    def test_final_pass_status_does_not_change_decision(self):
        self.create_draft()
        self.state["eval_result"] = {"passed": True, "items": {}, "action": "pass", "target_views": [], "feedback": []}
        result = self.agent.finalize(self.state)
        with pymupdf.open(result["report"]) as pdf:
            text = "\n".join(p.get_text() for p in pdf)
            self.assertIn("FINAL / PASS", text)
            self.assertIn("3.05 / 5.00", text)
            self.assertIn("조건부 검토", text)

    def test_finalize_requires_verdict(self):
        self.create_draft()
        with self.assertRaisesRegex(ValueError, "evaluator verdict"):
            self.agent.finalize(self.state)

    def test_finalize_rejects_string_pass_verdict(self):
        self.create_draft()
        self.state["eval_result"] = {"passed": "true"}
        with self.assertRaisesRegex(ValueError, "boolean passed"):
            self.agent.finalize(self.state)

    def test_same_run_rewrites_produce_different_pdf_paths(self):
        first = self.agent.run(self.state)
        self.state.update(first)
        second = self.agent.run(self.state)
        self.assertNotEqual(first["report"], second["report"])
        self.assertTrue(Path(first["report"]).is_file())

    def test_failed_view_and_missing_items_are_visible(self):
        self.state["analyses"]["market"].update(status="failed", missing_items=["시장 기준 시점"])
        self.state["node_errors"] = {"market": "검색 실패"}
        result = self.agent.run(self.state)
        notices = " ".join(x["text"] for x in result["report_draft"]["missing_information"])
        self.assertIn("시장 기준 시점", notices)
        self.assertIn("검색 실패", notices)
        self.assertIn("2개 이상", notices)

    @patch("dotenv.load_dotenv")
    @patch.dict("os.environ", {"OPENAI_MODEL": "test-model", "OPENAI_API_KEY": "test-key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_v2_eval_feedback_previous_draft_and_latest_research_reach_llm(self, model, _dotenv):
        self.state["report_draft"] = sample_draft().model_dump()
        self.state["eval_result"] = {"passed": False, "items": {}, "action": "rewrite", "target_views": [], "feedback": ["점수 근거 보완"]}
        runner = MagicMock()
        runner.invoke.return_value = sample_draft()
        model.return_value.with_structured_output.return_value = runner
        _write_report_draft(self.state)
        payload = json.loads(runner.invoke.call_args.args[0][1][1])
        self.assertEqual(payload["feedback"], ["점수 근거 보완"])
        self.assertEqual(payload["previous_draft"], self.state["report_draft"])
        self.assertEqual(payload["context"]["analyses"], self.state["analyses"])
        self.assertEqual(model.call_args.kwargs["temperature"], 0)

    def test_research_result_and_feedback_are_used_without_candidate_switch(self):
        from langgraph.graph import StateGraph, START, END
        builder = StateGraph(GraphState)
        seen = []
        def writer(state):
            seen.append(state["analyses"]["market"]["summary"])
            return sample_draft()
        self.agent.draft_writer = writer
        builder.add_node("report_generation", guard_node("report_generation", self.agent.run))
        def judge_double(state):
            failed = state["report_attempts"] == 0
            return {"eval_result": {"passed": not failed, "items": {}, "action": "research" if failed else "pass",
                                    "target_views": ["market"] if failed else [], "feedback": [],
                                    "research_hints": {"market": ["독립 근거 추가"]} if failed else {}},
                    "report_attempts": state["report_attempts"] + int(failed),
                    "resume_after_research": "report_generation"}
        def supervisor_double(state):
            return {"rework_hints": state["eval_result"]["research_hints"], "next_views": ["market"],
                    "eval_result": {**state["eval_result"], "target_views": [], "research_hints": {}}}
        def market_double(state):
            self.assertEqual(state["rework_hints"]["market"], ["독립 근거 추가"])
            return {"analyses": {"market": {**state["analyses"]["market"], "summary": "최신 재조사 결과"}}}
        builder.add_node("evaluator", guard_node("evaluator", judge_double))
        builder.add_node("supervisor", guard_node("supervisor", supervisor_double))
        builder.add_node("market", guard_node("market", market_double))
        builder.add_node("final_report", guard_node("report_generation", self.agent.finalize))
        builder.add_edge(START, "report_generation")
        builder.add_edge("report_generation", "evaluator")
        builder.add_conditional_edges("evaluator", lambda s: "final_report" if s["eval_result"]["passed"] else "supervisor")
        builder.add_edge("supervisor", "market")
        builder.add_edge("market", "report_generation")
        builder.add_edge("final_report", END)
        result = builder.compile().invoke(self.state)
        self.assertEqual(seen, ["검증 필요", "최신 재조사 결과"])
        self.assertEqual(result["selected_startup"], self.state["selected_startup"])
        self.assertEqual(result["report_attempts"], 1)
        self.assertEqual(result["resume_after_research"], "report_generation")
        self.assertTrue(Path(result["report"]).is_file())

    def test_langgraph_rewrite_then_final_evaluation_uses_real_report_node(self):
        from langgraph.graph import StateGraph, START, END
        builder = StateGraph(GraphState)
        seen = []
        def writer(state):
            seen.append((state.get("eval_result") or {}).get("feedback", []))
            return sample_draft()
        self.agent.draft_writer = writer
        builder.add_node("report_generation", guard_node("report_generation", self.agent.run))
        def judge_double(state):
            failed = state["report_attempts"] == 0
            return {"eval_result": {"passed": not failed, "items": {}, "action": "rewrite" if failed else "pass",
                                    "target_views": [], "feedback": ["리스크 균형 보완"] if failed else []},
                    "report_attempts": state["report_attempts"] + int(failed)}
        builder.add_node("evaluator", guard_node("evaluator", judge_double))
        builder.add_node("final_report", guard_node("report_generation", self.agent.finalize))
        builder.add_edge(START, "report_generation")
        builder.add_edge("report_generation", "evaluator")
        builder.add_conditional_edges("evaluator", lambda s: "final_report" if s["eval_result"]["passed"] else "report_generation")
        builder.add_edge("final_report", END)
        result = builder.compile().invoke(self.state)
        self.assertEqual(seen, [[], ["리스크 균형 보완"]])
        self.assertEqual(result["report_attempts"], 1)
        self.assertTrue(result["eval_result"]["passed"])
        self.assertTrue(Path(result["report"]).is_file())


if __name__ == "__main__":
    unittest.main()
