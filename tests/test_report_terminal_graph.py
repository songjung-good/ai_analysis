"""Regression tests against the assembled graph, real routing and Supervisor.

The evaluator and research responses are test doubles, not submission traces.
"""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image

from ai_investment.agents.report import ReportAgent
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.orchestration.supervisor import supervisor
from ai_investment.state import VIEWS
from test_report import sample_state, sample_draft


spec = importlib.util.spec_from_file_location("terminal_package", Path(__file__).resolve().parents[1] / "scripts/package_submission.py")
package_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_module)


def exercise_graph(directory, *, failures=3, research=False):
    state = sample_state()
    state["references"].append(Evidence(source_id="independent", title="외부 검증", url="https://independent.test/robot", excerpt="테스트용 검증 자료"))
    for view in VIEWS:
        state["analyses"][view].update(missing_items=[], source_ids=["used", "independent"])
    writer_calls, report_calls, eval_calls, worker_calls = [], [], [], []

    def writer(current):
        writer_calls.append(current["report_attempts"])
        if current["report_attempts"] >= 3 or (current.get("eval_result") or {}).get("passed"):
            raise AssertionError("A terminal report must reuse evaluated prose without an LLM")
        return sample_draft()

    agent = ReportAgent(output_dir=Path(directory), draft_writer=writer)

    def report(current):
        report_calls.append(copy.deepcopy(current))
        return agent.run(current)

    def evaluator(current):
        eval_calls.append(current["report_attempts"])
        failed = len(eval_calls) <= failures
        return {
            "eval_result": {
                "passed": not failed,
                "items": {"groundedness": {"passed": not failed, "reason": "시장 근거 부족" if failed else "검증 통과"}},
                "action": ("research" if research else "rewrite") if failed else "pass",
                "target_views": ["market"] if failed and research else [],
                "feedback": ["독립적인 출처를 보완하라"] if failed else [],
                "research_hints": {"market": ["독립적인 출처"]} if failed and research else {},
            },
            "report_attempts": current["report_attempts"] + int(failed),
            "resume_after_research": "report_generation",
        }

    def market(current):
        worker_calls.append("market")
        return {"analyses": {"market": copy.deepcopy(current["analyses"]["market"])}}

    def unexpected(current):
        raise AssertionError("Only the requested market view may be reworked")

    graph = build_graph(AgentNodes(
        startup_discovery=lambda current: {}, customer_profile=lambda current: {},
        technical_analysis=unexpected, business_analysis=unexpected, market_analysis=market,
        investment_decision=lambda current: {"decision": "conditional"},
        report_generation=report, supervisor=supervisor, stakeholder=unexpected, domain=unexpected,
        evaluator=evaluator,
    ))
    with patch("ai_investment.orchestration.supervisor.log_decision"):
        result = graph.invoke(state, config={"recursion_limit": 100})
    return result, writer_calls, report_calls, eval_calls, worker_calls


class TerminalReportGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def pdf_text(self, result):
        with pymupdf.open(result["report"]) as pdf:
            self.assertLessEqual(pdf.page_count, 10)
            return "\n".join(page.get_text() for page in pdf)

    def test_third_failure_outputs_limit_report_once_then_ends(self):
        result, writers, reports, evaluations, workers = exercise_graph(self.directory)
        self.assertEqual(writers, [0, 1, 2])
        self.assertEqual(evaluations, [0, 1, 2])
        self.assertEqual(len(reports), 4)
        self.assertEqual(workers, [])
        self.assertEqual(result["report_attempts"], 3)
        self.assertFalse(result["eval_result"]["passed"])
        for field in ("summary", *VIEWS, "risks", "decision_basis"):
            self.assertEqual(result["report_draft"][field], reports[-1]["report_draft"][field])
        text = self.pdf_text(result)
        for expected in ("FINAL / NOT PASSED", "품질 평가 미통과", "groundedness: 시장 근거 부족", "근거 부족 항목"):
            self.assertIn(expected, text)
        self.assertNotIn("DRAFT / EVALUATION PENDING", text)

    def test_first_pass_finalizes_once_without_re_evaluation(self):
        result, writers, reports, evaluations, workers = exercise_graph(self.directory, failures=0)
        self.assertEqual(writers, [0])
        self.assertEqual(evaluations, [0])
        self.assertEqual(len(reports), 2)
        self.assertEqual(result["report_attempts"], 0)
        self.assertTrue(result["eval_result"]["passed"])
        self.assertIn("FINAL / PASS", self.pdf_text(result))

    def test_failed_draft_is_rewritten_before_later_pass(self):
        result, writers, reports, evaluations, _ = exercise_graph(self.directory, failures=1)
        self.assertEqual(writers, [0, 1])
        self.assertEqual(len(reports), 3)
        self.assertEqual(evaluations, [0, 1])
        self.assertEqual(result["report_attempts"], 1)
        self.assertIn("FINAL / PASS", self.pdf_text(result))

    def test_research_failures_use_same_terminal_path_and_keep_candidate(self):
        result, writers, reports, evaluations, workers = exercise_graph(self.directory, research=True)
        self.assertEqual(workers, ["market", "market"])
        self.assertEqual(writers, [0, 1, 2])
        self.assertEqual(len(reports), 4)
        self.assertEqual(result["rework_count"]["market"], 2)
        self.assertEqual(result["report_attempts"], 3)
        self.assertEqual(result["selected_startup"]["name"], "가상로보틱스")
        self.assertIn("FINAL / NOT PASSED", self.pdf_text(result))

    def test_assembled_graph_outputs_can_be_packaged_for_both_terminal_verdicts(self):
        trace = self.directory / "test-trace.png"
        Image.new("RGB", (8, 8)).save(trace)  # Test fixture, never a real tracing artifact.
        for failures, name in ((0, "pass"), (3, "limit")):
            result, *_ = exercise_graph(self.directory / name, failures=failures)
            path = package_module.package_submission(
                repo_url="https://github.com/example/project", report=Path(result["report"]), traces=[trace],
                campus="TEST", class_name="TEST", names=name, run_id=result["trace_id"],
                output_dir=self.directory / "submission",
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
