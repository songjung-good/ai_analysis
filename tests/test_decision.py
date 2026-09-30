import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.decision import run
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.scoring import WEIGHTS, calculate_score
from ai_investment.state import create_initial_state


class FakeModel:
    def __init__(self, score=4, output=None):
        self.score = score
        self.output = output
        self.calls = 0
        self.payload = None

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.payload = json.loads(messages[1][1])
        if isinstance(self.output, Exception):
            raise self.output
        if self.output is not None:
            return self.output
        source = self.payload["evidence"][0]["source_id"]
        return assessment(self.score, source)


def assessment(score=4, source="current"):
    return {
        **{key: {"score": score, "reason": f"{key} 검증 근거", "source_ids": [source]}
           for key in WEIGHTS},
        "blocking_risks": [], "critical_information_gaps": [],
    }


def state():
    return {
        "selected_startup": {"name": "현재 기업"}, "profile": {},
        "technical_analysis": {"summary": "기술 확인", "source_ids": ["current"]},
        "business_analysis": {"summary": "유료 운영", "source_ids": ["current"]},
        "market_analysis": {"summary": "시장 확인", "source_ids": ["current"]},
        "references": [
            Evidence(source_id="previous", title="이전 기업", excerpt="이전 기업 근거"),
            Evidence(source_id="current", title="현재 기업", excerpt="항목별 근거를 가정한 대체 자료"),
        ],
        "evaluations": [{"startup_name": "이전 기업", "score": 2, "decision": "hold", "reason": "이전"}],
    }


class DecisionTests(unittest.TestCase):
    def test_thresholds_ownership_and_one_history_delta(self):
        for score, expected in ((4, "invest"), (3, "conditional"), (2, "hold")):
            with self.subTest(score=score):
                input_state = state()
                original = deepcopy(input_state)
                model = FakeModel(score)
                node = guard_node("investment_decision", lambda value: run(value, model=model))
                result = node(input_state)
                self.assertEqual(result["decision"], expected)
                self.assertAlmostEqual(result["investment_score"], score)
                self.assertEqual(len(result["evaluations"]), 1)
                self.assertEqual(result["evaluations"][0]["startup_name"], "현재 기업")
                self.assertEqual(set(result["evaluations"][0]["score_details"]), set(WEIGHTS))
                self.assertEqual([item["source_id"] for item in model.payload["evidence"]], ["current"])
                self.assertNotIn("evaluations", model.payload)
                self.assertEqual(input_state, original)

    def test_missing_analysis_or_evidence_skips_model(self):
        for field in ("technical_analysis", "business_analysis", "market_analysis", "sources"):
            with self.subTest(field=field):
                value = state()
                if field == "sources":
                    for key in ("technical_analysis", "business_analysis", "market_analysis"):
                        value[key].pop("source_ids")
                else:
                    value[field] = {}
                model = FakeModel()
                result = run(value, model=model)
                self.assertEqual(result["decision"], "hold")
                self.assertIsNone(result["investment_score"])
                self.assertEqual(model.calls, 0)

    def test_unknown_score_is_not_fabricated(self):
        output = assessment()
        output["investment_terms"] = {"score": None, "reason": "거래 조건 미공개", "source_ids": []}
        result = run(state(), model=FakeModel(output=output))
        self.assertEqual(result["decision"], "hold")
        self.assertIsNone(result["investment_score"])
        self.assertNotIn("investment_terms", result["scores"])
        self.assertIn("거래 조건 미공개", result["decision_reason"])

    def test_risk_and_critical_information_override_high_score(self):
        for override in (
            {"blocking_risks": [{"category": "safety", "reason": "출처로 확인된 중대한 안전 문제", "source_ids": ["current"]}]},
            {"critical_information_gaps": ["핵심 계약 내용 상충"]},
        ):
            with self.subTest(override=override):
                output = {**assessment(5), **override}
                result = run(state(), model=FakeModel(output=output))
                self.assertEqual(result["decision"], "hold")
                self.assertEqual(result["investment_score"], 5)

    def test_foreign_or_nonexistent_citations_fail(self):
        for source in ("previous", "invented"):
            with self.assertRaisesRegex(ValueError, "current source IDs"):
                run(state(), model=FakeModel(output=assessment(source=source)))
        value = state()
        value["technical_analysis"]["source_ids"] = ["missing"]
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "Analysis cites unavailable"):
            run(value, model=model)
        self.assertEqual(model.calls, 0)

    def test_invalid_scores_and_provider_failure_propagate(self):
        for score in (0, 6, float("nan"), True):
            with self.assertRaises(ValidationError):
                run(state(), model=FakeModel(output=assessment(score)))
        output = assessment()
        output["team"]["source_ids"] = []
        with self.assertRaises(ValidationError):
            run(state(), model=FakeModel(output=output))
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            run(state(), model=FakeModel(output=RuntimeError("model unavailable")))

    def test_invalid_name_fails(self):
        for name, exception in ((None, ValueError), (" ", ValueError), (3, TypeError)):
            value = state()
            value["selected_startup"]["name"] = name
            with self.assertRaises(exception):
                run(value, model=FakeModel())

    def test_rounding_does_not_promote_decision(self):
        result = calculate_score(dict.fromkeys(WEIGHTS, 3.999))
        self.assertEqual(result.decision, "conditional")
        result = calculate_score(dict.fromkeys(WEIGHTS, 2.999))
        self.assertEqual(result.decision, "hold")
        for value in (True, "4"):
            with self.assertRaises(ValueError):
                calculate_score(dict.fromkeys(WEIGHTS, value))

    def test_conditional_then_invest_graph_accumulates_once(self):
        candidates = [{"name": "A"}, {"name": "B"}]

        def analysis(field):
            def node(value):
                name = value["selected_startup"]["name"]
                return {
                    field: {"summary": name, "source_ids": [name]},
                    "references": [Evidence(source_id=name, title=name, excerpt=f"{name} 근거")],
                }
            return node

        def decide(value):
            name = value["selected_startup"]["name"]
            model = FakeModel(score=3 if name == "A" else 4)
            result = run(value, model=model)
            self.assertEqual({item["source_id"] for item in model.payload["evidence"]}, {name})
            return result

        graph = build_graph(AgentNodes(
            startup_discovery=lambda value: {"candidates": candidates, "current_idx": 0, "selected_startup": candidates[0]},
            customer_profile=lambda value: {"profile": {}},
            technical_analysis=analysis("technical_analysis"),
            business_analysis=analysis("business_analysis"),
            market_analysis=analysis("market_analysis"),
            investment_decision=decide,
            report_generation=lambda value: {"report": "보고서"},
        ))
        result = graph.invoke(create_initial_state(domain="Physical AI", criteria={}, max_iterations=2))
        self.assertEqual([item["decision"] for item in result["evaluations"]], ["conditional", "invest"])
        self.assertEqual([item["startup_name"] for item in result["evaluations"]], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
