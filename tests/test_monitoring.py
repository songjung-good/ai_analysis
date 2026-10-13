import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_investment.contracts import guard_node
from ai_investment.utils import logger as run_logger
from ai_investment.tools import web


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.trace = "monitor-test"
        patcher = patch.object(run_logger, "LOG_DIR", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)
        handler = run_logger.setup_logging(self.trace)

        def close():
            logging.getLogger("ai_investment").removeHandler(handler)
            handler.close()

        self.addCleanup(close)
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def records(self):
        path = self.directory / f"{self.trace}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def state(self, name="A", rework=0):
        return {"trace_id": self.trace, "selected_startup": {"name": name}, "rework_count": {"domain": rework}}

    def test_llm_timeout_is_saved_in_run_and_attempt_file_without_prompt(self):
        runnable = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(TimeoutError("fixture timeout")))
        model = SimpleNamespace(model_name="fixture-model", request_timeout=60, max_retries=0)

        def decision(state):
            run_logger.invoke_logged(runnable, "FULL_PROMPT_MUST_NOT_BE_SAVED", label="투자 평가 응답", model=model)

        with self.assertRaisesRegex(TimeoutError, "fixture timeout"):
            guard_node("investment_decision", decision)(self.state())
        records = self.records()
        self.assertEqual([record["event"] for record in records],
                         ["node_started", "llm_started", "llm_failed", "node_failed"])
        failure = records[2]
        self.assertEqual(failure["error_type"], "TimeoutError")
        self.assertEqual(failure["timeout_seconds"], 60)
        self.assertEqual(failure["sdk_max_retries"], 0)
        self.assertGreater(failure["input_chars"], 0)
        self.assertGreaterEqual(failure["elapsed_ms"], 0)
        self.assertEqual(len({record["attempt_id"] for record in records}), 1)
        self.assertNotIn("FULL_PROMPT_MUST_NOT_BE_SAVED", json.dumps(records))
        files = list((self.directory / self.trace / "attempts").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertEqual([json.loads(line) for line in files[0].read_text().splitlines()], records)
        self.assertIn("실패", self.output.getvalue())

    def test_returned_failure_is_not_printed_or_logged_as_success(self):
        result = {"analyses": {"domain": {"view": "domain", "status": "failed", "missing_items": []}},
                  "node_status": {"domain": "failed"}, "node_errors": {"domain": "TimeoutError: fixture"}}
        self.assertEqual(guard_node("domain", lambda _: result)(self.state()), result)
        self.assertEqual(self.records()[-1]["event"], "node_failed")
        self.assertEqual(self.records()[-1]["error"]["domain"], "TimeoutError: fixture")
        self.assertNotIn("완료", self.output.getvalue())

    def test_rework_and_parallel_async_nodes_have_separate_attempt_files(self):
        async def worker(state):
            with run_logger.monitor("retrieval", label="자료 확인"):
                await asyncio.sleep(0)
                run_logger.log_decision(state["trace_id"], "domain", "fixture", {"name": state["selected_startup"]["name"]})
            return {"analyses": {"domain": {"view": "domain", "status": "done", "missing_items": []}}}

        node = guard_node("domain", worker)

        async def execute():
            await asyncio.gather(node(self.state("A")), node(self.state("B", rework=1)))

        asyncio.run(execute())
        files = list((self.directory / self.trace / "attempts").glob("*.jsonl"))
        self.assertEqual(len(files), 2)
        for path in files:
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({record["target_name"] for record in records}), 1)
            self.assertEqual(len({record["attempt_id"] for record in records}), 1)
            name = records[0]["target_name"]
            self.assertTrue(all(record["node_attempt"] == (1 if name == "A" else 2) for record in records))
            self.assertTrue(any(record.get("event") == "routing_decision" for record in records))

    def test_log_viewer_displays_failures_and_unfinished_operations(self):
        path = self.directory / "viewer.jsonl"
        records = [
            {"event": "llm_started", "operation_id": "one", "operation": "llm", "ts": 1, "node": "decision"},
            {"event": "llm_failed", "operation_id": "one", "operation": "llm", "elapsed_ms": 60000,
             "status": "failed", "node": "decision", "error_type": "TimeoutError", "error": "fixture"},
            {"event": "search_started", "operation_id": "two", "operation": "search", "ts": 1, "node": "discovery"},
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"incomplete":', encoding="utf-8")
        script = Path(__file__).parents[1] / "scripts" / "inspect_run_log.py"
        result = subprocess.run([sys.executable, str(script), str(path), "--slowest", "2"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("실패 | 60.0초", result.stdout)
        self.assertIn("TimeoutError", result.stdout)
        self.assertIn("진행 중", result.stdout)

    def test_search_retry_and_successful_empty_result_are_distinguished(self):
        from langchain_core.tools import ToolException

        with patch("langchain_tavily.TavilySearch") as tool, \
             patch.object(web, "_next_search_at", 0), \
             patch.dict(os.environ, {"TAVILY_MIN_INTERVAL_SECONDS": "0", "TAVILY_RETRY_BACKOFF_SECONDS": "0"}):
            tool.return_value.invoke.side_effect = [
                ValueError("Error 429: fixture throttling"),
                ToolException("No search results found for 'fixture'. Suggestions: retry."),
            ]
            result = guard_node("domain", lambda _: {"references": web.web_search("fixture")})(self.state())
        self.assertEqual(result["references"], [])
        requests = [record for record in self.records() if record.get("event", "").startswith("web_request_")]
        self.assertEqual([record["event"] for record in requests],
                         ["web_request_started", "web_request_failed", "web_request_started", "web_request_finished"])
        self.assertEqual([record["attempt"] for record in requests], [1, 1, 2, 2])
        self.assertEqual(requests[-1]["status"], "completed")
        self.assertEqual(requests[-1]["result_count"], 0)
        self.assertEqual(len({record["operation_id"] for record in requests}), 2)
