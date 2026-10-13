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
from ai_investment.discover import load_candidate_names
from ai_investment.tools.web import _to_evidence, web_search


# Synthetic fixtures: these are not real companies or investment recommendations.
SOURCE = Evidence(
    source_id="official", title="Fictional company announcement",
    url="https://example.com/company", excerpt="Fixture: private independent robotics startup, Series A.",
)
UNUSED = Evidence(source_id="unused", title="Unrelated result", excerpt="Unrelated content")
CUSTOMER = Evidence(
    source_id="customer", title="Fictional customer deployment",
    url="https://customer.test/case", excerpt="Fixture: warehouse operator uses the robot in production.",
)


def fact(value, source="official"):
    return {"value": value, "source_ids": [source] if value is not None else []}


def check(confirmed=True):
    return {"confirmed": confirmed, "source_ids": ["official"] if confirmed is not None else []}


def verification(**updates):
    value = {
        "product": fact("물류창고에서 물품을 운반하는 AI 로봇"),
        "customer_application": fact("가상 물류회사가 창고에 도입해 운영 중", "customer"),
        "technical_evidence": check(),
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
    search = search if search is not None else Mock(return_value=[SOURCE, CUSTOMER, UNUSED])
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
        self.assertEqual(result["references"], [SOURCE, CUSTOMER])
        self.assertEqual(result["current_idx"], 0)
        self.assertEqual(search.call_count, 7)  # 초기 2회 + 후보 검증 5회(창업자 경력 포함)
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)

    def test_duplicate_company_and_candidate_limit(self):
        agent, _, search = agent_with([
            {"companies": [fact("Example Robotics"), fact("EXAMPLE-ROBOTICS"), fact("Second"), fact("Third")]},
            verification(), verification(),
        ])
        result = agent.run(state(candidate_limit=2))
        self.assertEqual([c["name"] for c in result["candidates"]], ["Example Robotics", "Second"])
        self.assertEqual(search.call_count, 12)
        self.assertEqual(result["references"], [SOURCE, CUSTOMER])

    def test_unknown_private_public_exited_wrong_domain_or_stage_are_excluded(self):
        for update in (
            {"private_company": check(None)}, {"private_company": check(False)},
            {"no_completed_exit": check(False)}, {"domain_match": check(False)},
            {"funding_stage": fact("Series D")}, {"funding_stage": fact(None)},
            {"product": fact(None)},
            {"customer_application": fact(None)},
            {"technical_evidence": check(None)},
            {"technical_evidence": check(False)},
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

    def test_truncated_citation_is_repaired_and_noise_ids_dropped(self):
        long_id = "official-announcement-0001"
        search = Mock(return_value=[SOURCE.model_copy(update={"source_id": long_id}), CUSTOMER])

        def relabel(value):
            if isinstance(value, dict):
                return {k: [long_id if sid == "official" else sid for sid in v] if k == "source_ids"
                        else relabel(v) for k, v in value.items()}
            return [relabel(item) for item in value] if isinstance(value, list) else value

        noisy = relabel(verification())
        noisy["product"] = {"value": "물류 로봇", "source_ids": ["official-ann", "bogus"]}
        agent, _, _ = agent_with([{"companies": [fact("Example Robotics", long_id)]}, noisy], search)
        result = agent.run(state())
        self.assertEqual(result["candidates"][0]["source_ids"], ["customer", long_id])

    def test_unknown_exit_of_confirmed_private_startup_is_presumed_no_exit(self):
        agent, _, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, verification(no_completed_exit=check(None)),
        ])
        self.assertEqual(len(agent.run(state())["candidates"]), 1)

    def test_candidate_with_repeatedly_invalid_verification_is_skipped(self):
        invented = verification(product=fact("물류 로봇", "invented"))
        agent, _, _ = agent_with([
            {"companies": [fact("Broken"), fact("Example Robotics")]},
            invented, invented, verification(),
        ])
        result = agent.run(state(candidate_limit=1))
        self.assertEqual([c["name"] for c in result["candidates"]], ["Example Robotics"])

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
            *[verification(private_company=check(False)) for _ in range(4)],
        ])
        self.assertEqual(agent.run(state(candidate_limit=1))["candidates"], [])
        self.assertEqual(search.call_count, 22)  # 초기 2회 + 최소 4개 후보 × 5회

    def test_existing_references_are_not_appended_again(self):
        agent, _, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, verification(),
        ])
        initial = state()
        initial["references"] = [SOURCE, CUSTOMER]
        self.assertEqual(agent.run(initial)["references"], [])

    def test_changed_search_excerpt_is_preserved_for_verification(self):
        new = SOURCE.model_copy(update={"excerpt": "A later funding announcement"})
        search = Mock(side_effect=[[SOURCE], [new], [], [], [CUSTOMER], [], []])
        agent, model, _ = agent_with([
            {"companies": [fact("Example Robotics")]}, verification(),
        ], search)
        result = agent.run(state())
        self.assertIn(SOURCE.excerpt, result["references"][0].excerpt)
        self.assertIn(new.excerpt, result["references"][0].excerpt)

    def test_bad_criteria_fail_before_search(self):
        for criteria in ({"candidate_limit": 0}, {"candidate_limit": 11}, {"funding_stages": []},
                         {"candidate_names": [" "]}, {"candidate_names": [123]}, {"typo": 1}):
            with self.subTest(criteria=criteria):
                agent, _, search = agent_with([])
                with self.assertRaises(ValueError):
                    agent.run(state(**criteria))
                search.assert_not_called()

    def test_shortlist_skips_broad_discovery_and_replaces_insufficient_candidate(self):
        agent, model, search = agent_with([
            verification(customer_application=fact(None)), verification(),
        ])
        with self.assertLogs("ai_investment.agents.discovery", level="INFO") as logs:
            result = agent.run(state(candidate_limit=1, candidate_names=["Sparse", "Ready", "Unused"]))
        self.assertEqual([item["name"] for item in result["candidates"]], ["Ready"])
        self.assertEqual(search.call_count, 10)
        self.assertTrue(all('"Sparse"' in call.args[0] or '"Ready"' in call.args[0]
                            for call in search.call_args_list))
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)
        self.assertIn("Sparse", logs.output[0])
        self.assertEqual(result["selected_startup"]["source_ids"], ["customer", "official"])

    def test_shortlist_with_no_search_results_skips_model_and_returns_no_candidates(self):
        agent, model, search = agent_with([], Mock(return_value=[]))
        result = agent.run(state(candidate_names=["Sparse"]))
        self.assertEqual(result["candidates"], [])
        model.with_structured_output.assert_not_called()
        self.assertEqual(search.call_count, 5)

    def test_uncited_sites_or_many_pages_on_same_site_do_not_meet_source_threshold(self):
        for customer in (CUSTOMER.model_copy(update={"url": "https://www.example.com/customer"}), None):
            with self.subTest(customer=customer):
                response = verification() if customer else verification(
                    customer_application=fact("창고에서 운영", "official"),
                )
                evidence = [SOURCE, customer] if customer else [SOURCE, CUSTOMER]
                agent, _, _ = agent_with([response], Mock(return_value=evidence))
                self.assertEqual(agent.run(state(candidate_names=["One site"]))["candidates"], [])

    def test_candidate_file_validates_names_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text('[" Ready ", "Other"]', encoding="utf-8")
            self.assertEqual(load_candidate_names(path), ["Ready", "Other"])
            for content in ('[]', '[" "]', '[123]', '{"names": ["Ready"]}', 'invalid JSON'):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_candidate_names(path)

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
    @patch("ai_investment.tools.web.sleep")
    def test_documented_zero_results_exception_is_empty_search(self, _sleep, tool):
        tool.return_value.invoke.side_effect = ToolException("No search results found for 'robotics'. Suggestions: retry.")
        self.assertEqual(web_search("robotics"), [])
        self.assertFalse(tool.call_args.kwargs["handle_tool_error"])

    @patch("langchain_tavily.TavilySearch")
    @patch("ai_investment.tools.web.sleep")
    def test_other_tool_errors_propagate(self, _sleep, tool):
        tool.return_value.invoke.side_effect = ToolException("Unexpected service error")
        with self.assertRaisesRegex(ToolException, "Unexpected service"):
            web_search("robotics")


if __name__ == "__main__":
    unittest.main()
