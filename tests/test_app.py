"""Exercise the production entrypoint/graph with fake research and Judge, real PDFs."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pymupdf

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import app
from ai_investment.agents.report import ReportAgent, ReportDraft
from ai_investment.agents.business_search import build_search_plan, collect_evidence
from ai_investment.models import Evidence
from ai_investment.orchestration.supervisor import supervisor
from ai_investment.utils import logger as run_logger


class AppIntegrationTests(unittest.TestCase):
    def execute(self, mode="pass", *, discover=True):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        calls = []
        sources = [
            Evidence(source_id="used", title="고객 자료", url="https://customer.test/case", excerpt="기업 현장 적용"),
            Evidence(source_id="other", title="외부 자료", url="https://research.test/case", excerpt="현장 제약 확인"),
        ]

        def discovery(_state):
            calls.append("discovery")
            if mode == "no_candidates":
                return {"candidates": []}
            return {"candidates": [{"name": "A"}, {"name": "B"}], "current_idx": 0,
                    "selected_startup": {"name": "A"}}

        def worker(state, *, view):
            calls.append(view)
            if view in {"stakeholder", "domain"}:
                collect_evidence(
                    build_search_plan(state, view=view), search=lambda *args, **kwargs: sources,
                    trace_id=state["trace_id"], view=view,
                    attempt=state["rework_count"][view], target_name=state["selected_startup"]["name"],
                )
            missing = ["운영 제약"] if mode == "exhausted" and view == "domain" else []
            analysis = {"view": view, "summary": "현장 적용과 제약을 확인했다.",
                        "claims": [{"text": "현장 적용과 제약", "source_ids": ["used", "other"]}],
                        "source_ids": ["used", "other"], "missing_items": missing, "status": "done"}
            update = {"analyses": {view: analysis}, "references": sources,
                      "node_status": {view: "done"}, "node_errors": {view: None}}
            if view in {"technical", "market"}:
                update[f"{view}_analysis"] = {"summary": analysis["summary"], "source_ids": analysis["source_ids"]}
            return update

        def decision(state):
            calls.append("decision")
            self.assertEqual(set(state["business_analysis"]), {"stakeholder", "domain"})
            name = state["selected_startup"]["name"]
            choice = "hold" if state.get("report_draft") else "invest"
            return {"decision": choice, "decision_reason": "Mock 점수 근거",
                    "scores": {}, "investment_score": None,
                    "evaluations": [{"startup_name": name, "score": None, "decision": choice, "reason": "Mock"}]}

        paragraph = [{"text": "현장 적용 사례와 조건을 확인했다.", "source_ids": ["used", "other"]}]
        draft = ReportDraft(**{section: paragraph for section in
                               ("summary", "company", "technical", "market", "stakeholder", "domain", "risks", "decision_basis")})
        agent = ReportAgent(output_dir=directory / "pdf", draft_writer=lambda _state: draft)

        def report(state):
            calls.append("report")
            return agent.run(state)

        def final(state):
            calls.append("final")
            return agent.finalize(state)

        def evaluate(state):
            calls.append("evaluate")
            research = mode in {"research", "exhausted", "no_candidates"} and state["report_attempts"] == 0
            failed = mode == "rewrite_limit" or research
            return {
                "eval_result": {"passed": not failed, "items": {"coverage": {"passed": not failed, "reason": "Mock 평가"}},
                                "action": "research" if research else "rewrite" if failed else "pass",
                                "target_views": ["market" if mode == "research" else "domain"] if research else [],
                                "feedback": [], "research_hints": {"market": ["독립 근거 추가"]} if research else {}},
                "report_attempts": state["report_attempts"] + int(failed),
                "resume_after_research": "investment_decision",
            }

        def coordinate(state):
            calls.append("supervisor")
            if calls.count("supervisor") == 2:
                self.assertEqual(set(state["analyses"]), {"technical", "market", "stakeholder", "domain"})
            return supervisor(state)

        argv = ["app.py", "--limit", "2", "--max-iterations", "2"]
        if not discover:
            argv.extend(["--company", "직접 지정 기업"])
        with patch.object(run_logger, "LOG_DIR", directory / "logs"), \
             patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}), \
             patch.object(app, "load_dotenv"), patch.object(sys, "argv", argv), \
             patch.multiple(app, startup_discovery=discovery, customer_profile=lambda state: {"profile": {}},
                            technical_analysis=lambda state: worker(state, view="technical"),
                            market_view=lambda state: worker(state, view="market"), business_analysis=worker,
                            investment_decision=decision, report_generation=report, finalize_report=final,
                            evaluate_report=evaluate, supervisor=coordinate), redirect_stdout(io.StringIO()):
            result = app.main()
        with pymupdf.open(result["report"]) as pdf:
            self.assertLessEqual(pdf.page_count, 10)
            text = "\n".join(page.get_text() for page in pdf)
        self.assertIn("SUMMARY", text)
        self.assertIn("REFERENCE", text)
        log = directory / "logs" / f"{result['trace_id']}.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(all(record["trace_id"] == result["trace_id"] for record in records))
        searches = [record for record in records if record.get("event") == "search_started"]
        if mode != "no_candidates":
            self.assertTrue(searches)
            self.assertTrue(all(record["query"] and record["target_name"] and "attempt" in record for record in searches))
        return result, calls, text

    def test_entrypoint_runs_four_views_and_finalizes_passed_pdf(self):
        result, calls, text = self.execute()
        self.assertTrue(result["eval_result"]["passed"])
        self.assertEqual(calls.count("supervisor"), 2)
        self.assertEqual(calls.count("final"), 1)
        self.assertIn("FINAL / PASS", text)

    def test_research_rescores_same_candidate_without_duplicate_history(self):
        result, calls, text = self.execute("research")
        self.assertEqual(result["selected_startup"]["name"], "A")
        self.assertEqual(calls.count("market"), 2)
        self.assertEqual(calls.count("decision"), 2)
        self.assertEqual(calls.count("report"), 2)
        self.assertEqual(result["evaluations"], [{"startup_name": "A", "score": None, "decision": "hold", "reason": "Mock"}])
        self.assertEqual(result["report_attempts"], 1)
        self.assertIn("FINAL / PASS", text)

    def test_rewrite_limit_finalizes_evaluated_text_without_fourth_generation(self):
        result, calls, text = self.execute("rewrite_limit")
        self.assertEqual(calls.count("report"), 3)
        self.assertEqual(calls.count("evaluate"), 3)
        self.assertEqual(result["report_attempts"], 3)
        self.assertIn("FINAL / NOT PASSED", text)

    def test_exhausted_view_stops_research_and_finalizes_failed_pdf(self):
        result, calls, text = self.execute("exhausted")
        self.assertEqual(calls.count("domain"), 3)
        self.assertEqual(calls.count("evaluate"), 1)
        self.assertEqual(calls.count("final"), 1)
        self.assertFalse(result["eval_result"]["passed"])
        self.assertIn("FINAL / NOT PASSED", text)

    def test_company_path_does_not_invent_product_or_funding_stage(self):
        result, calls, _text = self.execute(discover=False)
        self.assertNotIn("discovery", calls)
        self.assertEqual(result["selected_startup"], {"name": "직접 지정 기업"})

    def test_no_candidates_finalizes_without_dispatching_workers(self):
        result, calls, text = self.execute("no_candidates")
        self.assertNotIn("supervisor", calls)
        self.assertEqual(calls.count("evaluate"), 1)
        self.assertFalse(result["eval_result"]["passed"])
        self.assertIn("FINAL / NOT PASSED", text)

    def test_invalid_company_and_candidate_limit_rejected_before_execution(self):
        for args in (["--company", "  "], ["--limit", "0"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["app.py", *args]), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args()


if __name__ == "__main__":
    unittest.main()
