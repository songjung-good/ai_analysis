import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.tools import ToolException

from ai_investment.agents.discovery import DiscoveryAgent, run
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.state import create_initial_state
from ai_investment.tools.web import _to_evidence, web_search


# Synthetic fixtures: these are not real companies or investment recommendations.
SOURCE = Evidence(
    source_id="official", title="Fictional company announcement",
    url="https://example.com/company", excerpt="Fixture: private independent robotics startup, Series A.",
)
UNUSED = Evidence(source_id="unused", title="Unrelated result", excerpt="Unrelated content")


def fact(value, source="official"):
    return {"value": value, "source_ids": [source] if value is not None else []}


def check(confirmed=True):
    return {"confirmed": confirmed, "source_ids": ["official"] if confirmed is not None else []}


def verification(**updates):
    value = {
        "product": fact("물류창고에서 물품을 운반하는 AI 로봇"),
        "funding_stage": fact("Series A"),
        "team": [fact("가상 창업자")],
        "domain_match": check(), "region_match": check(),
        "private_company": check(), "no_completed_exit": check(),
        "disclosure_match": check(),
    }
    value.update(updates)
    return value


def state(**criteria):
    return create_initial_state(
        domain="Physical AI/Robotics", criteria=criteria, max_iterations=3,
    )


def agent_with(responses, search=None):
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = responses
    search = search if search is not None else Mock(return_value=[SOURCE, UNUSED])
    return DiscoveryAgent(llm=model, search=search, sleep=Mock()), model, search


class DiscoveryTests(unittest.TestCase):
    def test_success_returns_owned_keys_used_evidence_and_does_not_mutate_input(self):
        agent, model, search = agent_with([
            {"companies": [fact("Example Robotics — 물류용 로봇과 시리즈 A 투자 설명")]}, verification(),
        ])
        initial = state(candidate_limit=1)
        before = copy.deepcopy(initial)
        result = guard_node("startup_discovery", agent.run)(initial)
        self.assertEqual(initial, before)
        self.assertEqual(set(result), {"candidates", "current_idx", "selected_startup", "references"})
        self.assertEqual(result["selected_startup"], result["candidates"][0])
        self.assertEqual(result["selected_startup"]["name"], "Example Robotics")
        self.assertEqual(result["selected_startup"]["funding_stage"], "Series A")
        self.assertEqual(result["references"], [SOURCE])
        self.assertEqual(result["current_idx"], 0)
        self.assertEqual(search.call_count, 4)
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)

    def test_duplicate_company_and_candidate_limit(self):
        agent, _, search = agent_with([
            {"companies": [fact("Example Robotics"), fact("EXAMPLE-ROBOTICS"), fact("Second"), fact("Third")]},
            verification(), verification(),
        ])
        result = agent.run(state(candidate_limit=2))
        self.assertEqual([c["name"] for c in result["candidates"]], ["Example Robotics", "Second"])
        self.assertEqual(search.call_count, 6)
        self.assertEqual(result["references"], [SOURCE])

    def test_unknown_private_public_exited_wrong_domain_or_stage_are_excluded(self):
        for update in (
            {"private_company": check(None)}, {"private_company": check(False)},
            {"no_completed_exit": check(False)}, {"domain_match": check(False)},
            {"funding_stage": fact("Series D")}, {"funding_stage": fact(None)},
            {"product": fact(None)},
        ):
            with self.subTest(update=update):
                agent, _, _ = agent_with([
                    {"companies": [fact("Example Robotics")]}, verification(**update),
                ])
                result = agent.run(state())
                self.assertEqual(result["candidates"], [])
                self.assertEqual(result["selected_startup"], {})

    def test_region_disclosure_and_requested_funding_stage_are_enforced(self):
        cases = [
            ({"region": "대한민국"}, {"region_match": check(None)}),
            ({"disclosure_requirement": "공개된 고객 사례"}, {"disclosure_match": check(False)}),
            ({"funding_stages": ["Seed"]}, {}),
        ]
        for criteria, updates in cases:
            with self.subTest(criteria=criteria):
                agent, _, _ = agent_with([
                    {"companies": [fact("Example Robotics")]}, verification(**updates),
                ])
                self.assertEqual(agent.run(state(**criteria))["candidates"], [])

    def test_unknown_optional_team_is_not_invented(self):
        agent, _, _ = agent_with([
            {"companies": [fact("Example Robotics")]},
            verification(team=[], region_match=check(None), disclosure_match=check(None)),
        ])
        self.assertEqual(agent.run(state())["selected_startup"]["team"], [])

    def test_no_results_does_not_call_llm(self):
        agent, model, search = agent_with([], Mock(return_value=[]))
        self.assertEqual(agent.run(state())["candidates"], [])
        model.with_structured_output.assert_not_called()
        self.assertEqual(search.call_count, 2)

    def test_transient_search_failure_is_retried(self):
        search = Mock(side_effect=[TimeoutError("timeout"), [SOURCE], []])
        agent, _, _ = agent_with([{"companies": []}], search)
        self.assertEqual(agent.run(state())["candidates"], [])
        self.assertEqual(search.call_count, 3)
        agent.sleep.assert_called_once_with(0.5)

    def test_search_failure_is_not_hidden_as_empty_candidates(self):
        search = Mock(side_effect=TimeoutError("timeout"))
        agent, _, _ = agent_with([], search)
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            agent.run(state())
        self.assertEqual(search.call_count, 2)

    def test_invalid_search_configuration_is_not_retried(self):
        search = Mock(side_effect=ValueError("missing key"))
        agent, _, _ = agent_with([], search)
        with self.assertRaisesRegex(ValueError, "missing key"):
            agent.run(state())
        self.assertEqual(search.call_count, 1)

    def test_invented_citation_is_rejected_after_one_correction(self):
        invalid = {"companies": [fact("Invented", "not-returned-by-search")]}
        agent, model, search = agent_with([invalid, invalid])
        with self.assertRaisesRegex(ValueError, "invalid structured evidence"):
            agent.run(state())
        self.assertEqual(search.call_count, 2)
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)

    def test_uncited_eligibility_can_be_corrected_once(self):
        invalid = verification(private_company={"confirmed": True, "source_ids": []})
        agent, _, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, invalid, verification(),
        ])
        self.assertEqual(len(agent.run(state())["candidates"]), 1)

    def test_proposal_budget_bounds_searches_even_if_model_ignores_limit(self):
        agent, _, search = agent_with([
            {"companies": [fact(f"Company {i}") for i in range(10)]},
            verification(private_company=check(False)),
            verification(private_company=check(False)),
        ])
        self.assertEqual(agent.run(state(candidate_limit=1))["candidates"], [])
        self.assertEqual(search.call_count, 6)

    def test_existing_references_are_not_appended_again(self):
        agent, _, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, verification(),
        ])
        initial = state()
        initial["references"] = [SOURCE]
        self.assertEqual(agent.run(initial)["references"], [])

    def test_changed_search_excerpt_is_preserved_for_verification(self):
        new = SOURCE.model_copy(update={"excerpt": "A later funding announcement"})
        search = Mock(side_effect=[[SOURCE], [new], [], []])
        agent, model, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, verification(),
        ], search)
        result = agent.run(state())
        self.assertIn(SOURCE.excerpt, result["references"][0].excerpt)
        self.assertIn(new.excerpt, result["references"][0].excerpt)

    def test_bad_criteria_fail_before_search(self):
        for criteria in ({"candidate_limit": 0}, {"candidate_limit": 11}, {"funding_stages": []}, {"typo": 1}):
            with self.subTest(criteria=criteria):
                agent, _, search = agent_with([])
                with self.assertRaises(ValueError):
                    agent.run(state(**criteria))
                search.assert_not_called()

    @patch("dotenv.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_environment_has_actionable_error(self, _load):
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY, OPENAI_MODEL, TAVILY_API_KEY"):
            run(state())


class EmptyCandidateGraphTests(unittest.TestCase):
    def test_empty_discovery_routes_directly_to_report(self):
        agent, _, _ = agent_with([], Mock(return_value=[]))

        def unexpected(_state):
            self.fail("Analysis/decision nodes must not run without a candidate")

        def report(current):
            self.assertEqual(current["candidates"], [])
            self.assertEqual(current["evaluations"], [])
            return {"report": "선정 조건을 확인할 수 있는 후보가 없습니다."}

        graph = build_graph(AgentNodes(
            agent.run, unexpected, unexpected, unexpected, unexpected, unexpected, report,
        ))
        self.assertIn("후보가 없습니다", graph.invoke(state())["report"])


class DirectExecutionTests(unittest.TestCase):
    def test_script_path_works_outside_repository_without_pythonpath(self):
        script = Path(__file__).resolve().parents[1] / "src/ai_investment/agents/discovery.py"
        env = {**os.environ, "PYTHONPATH": ""}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"], cwd=directory, env=env,
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("--region", result.stdout)
        self.assertIn("--limit", result.stdout)


class WebFailureTests(unittest.TestCase):
    def test_error_payload_does_not_become_empty_evidence(self):
        error = TimeoutError("service timeout")
        with self.assertRaises(TimeoutError) as raised:
            _to_evidence({"error": error}, 5)
        self.assertIs(raised.exception, error)

    def test_invalid_payload_is_not_an_empty_search(self):
        for payload in ("API error", {}, {"results": None}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                _to_evidence(payload, 5)

    @patch("langchain_tavily.TavilySearch")
    def test_documented_zero_results_exception_is_empty_search(self, tool):
        tool.return_value.invoke.side_effect = ToolException("No search results found for 'robotics'. Suggestions: retry.")
        self.assertEqual(web_search("robotics"), [])
        self.assertFalse(tool.call_args.kwargs["handle_tool_error"])

    @patch("langchain_tavily.TavilySearch")
    def test_other_tool_errors_propagate(self, tool):
        tool.return_value.invoke.side_effect = ToolException("Unexpected service error")
        with self.assertRaisesRegex(ToolException, "Unexpected service"):
            web_search("robotics")


if __name__ == "__main__":
    unittest.main()
