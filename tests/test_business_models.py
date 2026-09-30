import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business_models import BusinessAnalysis, CustomerCase, DeploymentCost, DeploymentEffect


class BusinessModelTests(unittest.TestCase):
    def test_confirmed_paid_result_and_missing_customer_rejection(self):
        values = dict(
            commercialization_stage="paid_operation", stage_reason="유료 현장 운영",
            stage_source_ids=["source-1"],
            customer_cases=[dict(customer_name="가상 고객", use_case="조립",
                                 operation_stage="paid_operation", is_paid=True,
                                 source_ids=["source-1"])],
            deployment_costs=[], deployment_effects=[], regulatory_risks=[],
            evidence_assessment=[], information_gaps=["가격 미공개"],
            summary="해당 고객의 유료 운영이 확인됐다.",
        )
        self.assertEqual(BusinessAnalysis(**values).model_dump()["customer_cases"][0]["is_paid"], True)
        with self.assertRaises(ValidationError):
            BusinessAnalysis(**{**values, "customer_cases": []})

    def test_unknown_result_serializes_without_inventing_evidence(self):
        result = BusinessAnalysis(
            commercialization_stage="unknown",
            stage_reason="운영 근거를 확인하지 못했다.",
            stage_source_ids=[],
            customer_cases=[],
            deployment_costs=[],
            deployment_effects=[],
            regulatory_risks=[],
            evidence_assessment=[],
            information_gaps=["고객 현장 운영 여부 확인 필요"],
            summary="공개 근거가 부족하다.",
        )
        self.assertEqual(result.model_dump()["commercialization_stage"], "unknown")
        self.assertEqual(result.model_dump()["customer_cases"], [])

    def test_paid_pilot_is_not_paid_operation(self):
        case = CustomerCase(
            customer_name="가상 고객", use_case="조립 시험",
            operation_stage="pilot", is_paid=True, source_ids=["source-1"],
        )
        self.assertEqual(case.operation_stage, "pilot")
        with self.assertRaises(ValidationError):
            CustomerCase(**{**case.model_dump(), "operation_stage": "paid_operation", "is_paid": None})

    def test_rejects_wrong_type_and_empty_source(self):
        for override in ({"is_paid": "yes"}, {"source_ids": [" "]}):
            with self.assertRaises(ValidationError):
                CustomerCase(**{
                    "customer_name": "가상 고객", "use_case": "조립",
                    "operation_stage": "pilot", "is_paid": None,
                    "source_ids": ["source-1"], **override,
                })

    def test_cost_accepts_unknown_amount_but_known_amount_needs_currency(self):
        values = dict(cost_type="purchase", description="가격 미공개", amount=None,
                      currency=None, period=None, scope="로봇 본체", source_ids=["source-1"])
        self.assertIsNone(DeploymentCost(**values).amount)
        with self.assertRaises(ValidationError):
            DeploymentCost(**{**values, "amount": 100})

    def test_numeric_effect_requires_unit(self):
        with self.assertRaises(ValidationError):
            DeploymentEffect(metric="생산성", value=20, unit=None, baseline=None,
                             measurement_conditions=None, description="기업 발표",
                             source_ids=["source-1"])


if __name__ == "__main__":
    unittest.main()
