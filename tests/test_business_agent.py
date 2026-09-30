import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business import run
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.state import create_initial_state


class FakeModel:
    def __init__(self, source="used", output=None):
        self.source = source
        self.output = output
        self.calls = 0
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        if isinstance(self.output, Exception):
            raise self.output
        if self.output is not None:
            return self.output
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

    def output(self, stage, sources, *, paid=None):
        """Canned responses test the output contract, not live LLM reasoning."""
        return dict(
            commercialization_stage=stage, stage_reason="제공된 사례에 한정한 단계 판단",
            stage_source_ids=sources,
            customer_cases=[dict(
                customer_name="가상 공장", use_case="조립 자동화",
                operation_stage=stage, is_paid=paid, source_ids=sources,
            )] if stage != "unknown" else [],
            deployment_costs=[], deployment_effects=[], regulatory_risks=[],
            evidence_assessment=[], information_gaps=["도입 가격 확인 필요"],
            summary="가격·현장 효과의 추가 확인이 필요하다.",
        )

    def execute(self, output, evidence):
        model = FakeModel(output=output)
        result = run(
            {"selected_startup": {"name": "가상 기업"}},
            search=lambda *args, **kwargs: evidence, model=model,
        )
        self.assertEqual(model.calls, 1)
        return result, model

    def test_demo_paid_operation_and_paid_pilot_remain_distinct(self):
        cases = [
            ("demo", None, "전시회 시연과 파트너십 발표. 고객 운영은 확인되지 않음."),
            ("paid_operation", True, "고객 현장에서 유료 계약으로 로봇 운영 중."),
            ("pilot", True, "고객 현장에서 유료 PoC 수행 중. 상시 운영은 아직 아님."),
        ]
        for stage, paid, excerpt in cases:
            with self.subTest(stage=stage):
                evidence = [Evidence(source_id="case", title="고객 사례", excerpt=excerpt)]
                result, model = self.execute(self.output(stage, ["case"], paid=paid), evidence)
                analysis = result["business_analysis"]
                self.assertEqual(analysis["commercialization_stage"], stage)
                self.assertEqual(analysis["customer_cases"][0]["is_paid"], paid)
                self.assertEqual([item.source_id for item in result["references"]], ["case"])
                self.assertIn(excerpt, model.messages[1][1])

    def test_undisclosed_cost_and_qualitative_effect_keep_null_values(self):
        output = self.output("pilot", ["case"])
        output["deployment_costs"] = [dict(
            cost_type="purchase", description="제품 가격 미공개", amount=None,
            currency=None, period=None, scope="로봇 본체", source_ids=["case"],
        )]
        output["deployment_effects"] = [dict(
            metric="생산성", value=None, unit=None, baseline=None,
            measurement_conditions=None, description="기업의 정성적 개선 주장",
            source_ids=["case"],
        )]
        output["evidence_assessment"] = [dict(
            claim="생산성 개선", level="claim_only", reason="수치·측정 조건 미공개",
            source_ids=["case"],
        )]
        result, _ = self.execute(output, [Evidence(
            source_id="case", title="기업 발표", excerpt="시험 운영 중 생산성 개선. 가격과 수치는 미공개."
        )])
        analysis = result["business_analysis"]
        self.assertIsNone(analysis["deployment_costs"][0]["amount"])
        self.assertIsNone(analysis["deployment_effects"][0]["value"])
        self.assertEqual(analysis["evidence_assessment"][0]["level"], "claim_only")

    def test_conflicting_sources_preserved_with_unknown_stage(self):
        output = self.output("unknown", ["company", "customer"])
        output["stage_reason"] = "기업과 고객의 운영 상태 발표가 상충한다."
        output["evidence_assessment"] = [dict(
            claim="고객 현장의 상시 운영 여부", level="conflicting",
            reason="기업은 운영을 발표했으나 고객은 시험 종료를 발표했다.",
            source_ids=["company", "customer"],
        )]
        output["information_gaps"].append("현재 계약과 운영 상태 재확인 필요")
        result, _ = self.execute(output, [
            Evidence(source_id="company", title="기업 발표", excerpt="현재 고객 공장에서 운영 중."),
            Evidence(source_id="customer", title="고객 발표", excerpt="시험 종료 후 현재 운영하지 않음."),
        ])
        analysis = result["business_analysis"]
        self.assertEqual(analysis["commercialization_stage"], "unknown")
        self.assertEqual(analysis["evidence_assessment"][0]["level"], "conflicting")
        self.assertEqual({item.source_id for item in result["references"]}, {"company", "customer"})

    def test_invalid_paid_operation_and_model_failure_propagate(self):
        evidence = [Evidence(source_id="case", title="기업 발표", excerpt="시연만 공개")]
        invalid = self.output("paid_operation", ["case"], paid=None)
        with self.assertRaises(ValidationError):
            self.execute(invalid, evidence)
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self.execute(RuntimeError("model unavailable"), evidence)

    def test_graph_candidate_loop_keeps_business_data_separate(self):
        snapshots = []
        candidates = [{"name": "가상 A"}, {"name": "가상 B"}]

        def business_node(state):
            name = state["selected_startup"]["name"]
            output = self.output("pilot", [name])
            output["summary"] = f"{name} 분석"
            model = FakeModel(output=output)
            result = run(state, search=lambda *args, **kwargs: [
                Evidence(source_id=name, title=f"{name} 자료", excerpt=f"{name} 시험 운영")
            ], model=model)
            payload = json.loads(model.messages[1][1])
            self.assertEqual(payload["selected_startup"]["name"], name)
            self.assertEqual([item["source_id"] for item in payload["evidence"]], [name])
            return result

        def decision(state):
            name = state["selected_startup"]["name"]
            snapshots.append((name, state["business_analysis"]["summary"]))
            return {"decision": "hold"}

        graph = build_graph(AgentNodes(
            startup_discovery=lambda state: {
                "candidates": candidates, "current_idx": 0, "selected_startup": candidates[0],
            },
            customer_profile=lambda state: {"profile": {}},
            technical_analysis=lambda state: {"technical_analysis": {}},
            business_analysis=business_node,
            market_analysis=lambda state: {"market_analysis": {}},
            investment_decision=decision,
            report_generation=lambda state: {"report": "모든 후보 보류"},
        ))
        result = graph.invoke(create_initial_state(domain="Physical AI", criteria={}, max_iterations=2))
        self.assertEqual(snapshots, [("가상 A", "가상 A 분석"), ("가상 B", "가상 B 분석")])
        self.assertEqual(result["report"], "모든 후보 보류")
        self.assertEqual([item.source_id for item in result["references"]], ["가상 A", "가상 B"])

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
