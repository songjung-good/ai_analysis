import sys
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.tools.web import _to_evidence
from ai_investment.tools import web


class WebErrorTests(unittest.TestCase):
    def test_provider_error_is_not_an_empty_success(self):
        with self.assertRaises(ConnectionError):
            _to_evidence({"error": ConnectionError("connection unavailable")}, 5)
        with self.assertRaisesRegex(RuntimeError, "Tavily search returned an error"):
            _to_evidence({"error": "provider failure"}, 5)


class WebRateLimitTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.sleeps = []

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.now += seconds

        for index, replacement in enumerate((
            patch("langchain_tavily.TavilySearch"),
            patch.object(web, "monotonic", side_effect=lambda: self.now),
            patch.object(web, "sleep", side_effect=sleep),
            patch.object(web, "_next_search_at", 0.0),
            patch.object(web, "logger"),
            patch.dict(os.environ, {"TAVILY_MIN_INTERVAL_SECONDS": "1", "TAVILY_RETRY_BACKOFF_SECONDS": "15"}),
        )):
            mocked = replacement.start()
            self.addCleanup(replacement.stop)
            if index == 0:
                self.tool = mocked.return_value
        self.success = {"results": [{"title": "Source", "url": "https://example.com", "content": "Evidence"}]}

    def test_sdk_valueerror_429_retries_after_backoff_then_returns_evidence(self):
        self.tool.invoke.side_effect = [{"error": ValueError("Error 429: excessive requests")}, self.success]
        result = web.web_search("robotics")
        self.assertEqual(len(result), 1)
        self.assertEqual(self.tool.invoke.call_count, 2)
        self.assertEqual(self.sleeps, [15.0])

    def test_persistent_429_stops_after_three_calls_without_discovery_retry(self):
        from ai_investment.agents.discovery import DiscoveryAgent

        self.tool.invoke.return_value = {"error": ValueError("Error 429: excessive requests")}
        outer_sleep = Mock()
        discovery = DiscoveryAgent(llm=Mock(), search=web.web_search, sleep=outer_sleep)
        with self.assertRaisesRegex(web.TavilyRateLimitError, "after 3 attempts"):
            discovery._search("robotics")
        self.assertEqual(self.tool.invoke.call_count, 3)
        self.assertEqual(self.sleeps, [15.0, 30.0])
        outer_sleep.assert_not_called()

    def test_other_provider_errors_are_not_retried_or_masked(self):
        error = ValueError("Error 401: invalid API key")
        self.tool.invoke.return_value = {"error": error}
        with self.assertRaises(ValueError) as raised:
            web.web_search("robotics")
        self.assertIs(raised.exception, error)
        self.assertEqual(self.tool.invoke.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_different_agent_tools_share_the_same_request_interval(self):
        from ai_investment.tools.registry import tools_for

        called_at = []

        def invoke(_input):
            called_at.append(self.now)
            return self.success

        self.tool.invoke.side_effect = invoke
        tools_for("business_analysis")[0]("business")
        tools_for("technical_analysis")[-1]("technical")
        self.assertEqual(called_at, [0.0, 1.0])

    def test_string_error_payload_429_keeps_status_and_is_retried(self):
        self.tool.invoke.side_effect = [{"error": "Error 429: excessive requests"}, self.success]
        self.assertTrue(web.web_search("robotics"))
        self.assertEqual(self.tool.invoke.call_count, 2)

    def test_invalid_pacing_config_does_not_invoke_provider(self):
        for value in ("-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"TAVILY_MIN_INTERVAL_SECONDS": value}):
                with self.assertRaises(ValueError):
                    web.web_search("robotics")
        self.tool.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
