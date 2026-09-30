import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.contracts import guard_node
from ai_investment.control import route_after_decision, select_next_candidate
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.scoring import WEIGHTS, calculate_score
from ai_investment.state import create_initial_state
from ai_investment.tools.registry import tools_for
from ai_investment.tools.web import _to_evidence


class InitialStateTests(unittest.TestCase):
    def test_rejects_invalid_iteration_limit(self):
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            create_initial_state(domain="Physical AI", criteria={}, max_iterations=0)


class OwnershipTests(unittest.TestCase):
    def test_rejects_cross_agent_write(self):
        guarded = guard_node(
            "technical_analysis",
            lambda _state: {"market_analysis": {"wrong_owner": True}},
        )
        with self.assertRaisesRegex(ValueError, "market_analysis"):
            guarded({})

    def test_allows_owned_field_and_shared_reducer_field(self):
        guarded = guard_node(
            "technical_analysis",
            lambda _state: {"technical_analysis": {}, "references": []},
        )
        self.assertEqual(
            guarded({}), {"technical_analysis": {}, "references": []}
        )


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "candidates": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "current_idx": 0,
            "max_iterations": 3,
        }

    def test_investment_routes_to_report(self):
        self.state["decision"] = "invest"
        self.assertEqual(route_after_decision(self.state), "report_generation")

    def test_hold_routes_to_next_candidate(self):
        self.state["decision"] = "hold"
        self.assertEqual(route_after_decision(self.state), "next_candidate")

    def test_conditional_routes_to_next_candidate(self):
        self.state["decision"] = "conditional"
        self.assertEqual(route_after_decision(self.state), "next_candidate")

    def test_iteration_limit_routes_to_report(self):
        self.state.update(decision="hold", current_idx=1, max_iterations=2)
        self.assertEqual(route_after_decision(self.state), "report_generation")

    def test_candidate_exhaustion_routes_to_report(self):
        self.state.update(decision="hold", current_idx=2)
        self.assertEqual(route_after_decision(self.state), "report_generation")

    def test_selects_next_candidate(self):
        self.assertEqual(
            select_next_candidate(self.state),
            {"current_idx": 1, "selected_startup": {"name": "B"}},
        )


class GraphIntegrationTests(unittest.TestCase):
    def test_hold_then_invest_waits_for_parallel_agents(self):
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.skipTest("langgraph is not installed")

        calls = []

        def discovery(_state):
            candidates = [{"name": "A"}, {"name": "B"}]
            return {
                "candidates": candidates,
                "current_idx": 0,
                "selected_startup": candidates[0],
                "references": [],
            }

        def profile(state):
            calls.append(("profile", state["selected_startup"]["name"]))
            return {"profile": {}, "references": []}

        def analysis(field):
            def run(state):
                calls.append((field, state["selected_startup"]["name"]))
                return {field: {}, "references": []}

            return run

        def decision(state):
            name = state["selected_startup"]["name"]
            calls.append(("decision", name))
            choice = "invest" if name == "B" else "hold"
            return {
                "scores": {},
                "investment_score": 4.0,
                "decision": choice,
                "decision_reason": "test",
                "evaluations": [
                    {
                        "startup_name": name,
                        "score": 4.0,
                        "decision": choice,
                        "reason": "test",
                    }
                ],
            }

        graph = build_graph(
            AgentNodes(
                discovery,
                profile,
                analysis("technical_analysis"),
                analysis("business_analysis"),
                analysis("market_analysis"),
                decision,
                lambda state: {"report": state["selected_startup"]["name"]},
            )
        )
        result = graph.invoke(
            create_initial_state(
                domain="Physical AI/Robotics", criteria={}, max_iterations=2
            )
        )

        self.assertEqual(result["report"], "B")
        self.assertEqual(
            [item["startup_name"] for item in result["evaluations"]], ["A", "B"]
        )
        for startup in ("A", "B"):
            decision_idx = calls.index(("decision", startup))
            for field in (
                "technical_analysis",
                "business_analysis",
                "market_analysis",
            ):
                self.assertLess(calls.index((field, startup)), decision_idx)


class ScoringTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(calculate_score(dict.fromkeys(WEIGHTS, 4)).decision, "invest")
        self.assertEqual(
            calculate_score(dict.fromkeys(WEIGHTS, 3)).decision, "conditional"
        )
        self.assertEqual(calculate_score(dict.fromkeys(WEIGHTS, 2)).decision, "hold")

    def test_missing_score_forces_hold_without_inventing_value(self):
        result = calculate_score({"team": 4})
        self.assertEqual(result.decision, "hold")
        self.assertIsNone(result.weighted_score)
        self.assertIn("market", result.missing_fields)

    def test_rejects_out_of_range_score(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            calculate_score(dict.fromkeys(WEIGHTS, 6))


class ToolTests(unittest.TestCase):
    def test_agent_receives_only_authorized_tools(self):
        self.assertEqual(
            [tool.__name__ for tool in tools_for("technical_analysis")],
            ["search_tech_docs", "web_search"],
        )
        self.assertEqual(tools_for("customer_profile"), ())

    def test_tavily_result_uses_common_evidence_model(self):
        results = _to_evidence(
            {
                "results": [
                    {
                        "title": "Source",
                        "url": "https://example.com",
                        "content": "Evidence text",
                        "published_date": "2026-01-01",
                    }
                ]
            },
            5,
        )
        self.assertEqual(
            results,
            [
                Evidence(
                    source_id=results[0].source_id,
                    title="Source",
                    url="https://example.com",
                    excerpt="Evidence text",
                    published_at="2026-01-01",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
