import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business import run
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.state import create_initial_state


class FakeModel:
    def __init__(self, source="used"):
        self.source = source
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        return dict(
            commercialization_stage="pilot", stage_reason="현장 시험 운영",
            stage_source_ids=[self.source], customer_cases=[], deployment_costs=[],
            deployment_effects=[], regulatory_risks=[], evidence_assessment=[],
            information_gaps=[], summary="시범 도입 근거가 있다.",
        )


def fake_search(*args, **kwargs):
    return [
        Evidence(source_id="used", title="기업 발표", excerpt="고객 현장 시험 운영"),
        Evidence(source_id="unused", title="기타", excerpt="다른 자료"),
    ]


class BusinessAgentTests(unittest.TestCase):
    def test_returns_only_owned_keys_and_cited_new_references(self):
        state = {
            "selected_startup": {"name": "가상 기업"},
            "references": [Evidence(source_id="previous", title="기존", excerpt="기존 자료")],
            "business_analysis": {"summary": "이전 기업 분석"},
            "technical_analysis": {"test": True},
        }
        original = deepcopy(state)
        model = FakeModel()
        node = guard_node("business_analysis", lambda value: run(value, search=fake_search, model=model))
        result = node(state)
        self.assertEqual(set(result), {"business_analysis", "references"})
        self.assertEqual([item.source_id for item in result["references"]], ["used"])
        self.assertEqual(result["business_analysis"]["commercialization_stage"], "pilot")
        self.assertIn("입력 누락: selected_startup.product", result["business_analysis"]["information_gaps"])
        self.assertEqual(model.calls, 1)
        self.assertEqual(state, original)

    def test_empty_search_skips_llm_and_returns_unknown(self):
        model = FakeModel()
        result = run({"selected_startup": {"name": "기업"}}, search=lambda *a, **kw: [], model=model)
        self.assertEqual(result["business_analysis"]["commercialization_stage"], "unknown")
        self.assertEqual(result["references"], [])
        self.assertEqual(model.calls, 0)

    def test_invalid_input_does_not_search(self):
        def unexpected_search(*args, **kwargs):
            self.fail("search must not run on invalid input")

        with self.assertRaises(ValueError):
            run({}, search=unexpected_search, model=FakeModel())

    def test_search_failure_propagates_without_llm(self):
        model = FakeModel()

        def failing_search(*args, **kwargs):
            raise RuntimeError("search failed")

        with self.assertRaisesRegex(RuntimeError, "search failed"):
            run({"selected_startup": {"name": "기업"}}, search=failing_search, model=model)
        self.assertEqual(model.calls, 0)

    def test_invalid_citation_is_not_returned_as_success(self):
        with self.assertRaisesRegex(ValueError, "Unknown evidence source IDs"):
            run({"selected_startup": {"name": "기업"}}, search=fake_search, model=FakeModel("invented"))

    def test_real_business_node_in_parallel_graph(self):
        def decision(state):
            self.assertIn("technical_analysis", state)
            self.assertIn("market_analysis", state)
            self.assertEqual(state["business_analysis"]["commercialization_stage"], "pilot")
            return {"decision": "invest"}

        graph = build_graph(AgentNodes(
            startup_discovery=lambda state: {
                "candidates": [{"name": "기업"}], "current_idx": 0,
                "selected_startup": {"name": "기업"},
            },
            customer_profile=lambda state: {"profile": {}},
            technical_analysis=lambda state: {"technical_analysis": {}},
            business_analysis=lambda state: run(state, search=fake_search, model=FakeModel()),
            market_analysis=lambda state: {"market_analysis": {}},
            investment_decision=decision,
            report_generation=lambda state: {"report": "통합 검증"},
        ))
        result = graph.invoke(create_initial_state(domain="Physical AI", criteria={}, max_iterations=1))
        self.assertEqual(result["report"], "통합 검증")
        self.assertEqual([item.source_id for item in result["references"]], ["used"])


if __name__ == "__main__":
    unittest.main()
