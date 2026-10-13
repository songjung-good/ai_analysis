import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business import run
from ai_investment.agents.business_analysis import analyze_business
from ai_investment.agents.business_search import build_search_plan
from ai_investment.models import Evidence


class FakeModel:
    def __init__(self):
        self.messages = None
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        self.calls += 1
        return {
            "commercialization_stage": "unknown", "stage_reason": "현장 단계 미확인",
            "stage_source_ids": [], "customer_cases": [], "deployment_costs": [],
            "deployment_effects": [], "regulatory_risks": [],
            "evidence_assessment": [{
                "claim": "고객이 설치 비용을 부담한다.", "level": "single_source",
                "reason": "고객 자료", "source_ids": ["customer"],
            }],
            "information_gaps": [], "summary": "고객의 설치 비용 부담이 확인됐다.",
        }


class BusinessViewTests(unittest.TestCase):
    def test_view_reaches_queries_prompt_and_payload_without_state_mutation(self):
        state = {"selected_startup": {"name": "가상 기업"}}
        original = deepcopy(state)
        queries = {}
        for view, instruction in (
            ("stakeholder", "이해 충돌"), ("domain", "운영 제약"),
        ):
            with self.subTest(view=view):
                calls = []

                def search(query, **kwargs):
                    calls.append(query)
                    return [Evidence(
                        source_id="customer", title="고객 자료", excerpt="고객이 설치 비용 부담",
                    )]

                model = FakeModel()
                result = run(state, view=view, search=search, model=model)
                queries[view] = calls
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(query.startswith("가상 기업 ") for query in calls))
                self.assertIn(instruction, model.messages[0][1])
                self.assertEqual(json.loads(model.messages[1][1])["view"], view)
                self.assertEqual([item.source_id for item in result["references"]], ["customer"])
                self.assertEqual(state, original)
        self.assertNotEqual(queries["stakeholder"], queries["domain"])

    def test_invalid_view_fails_before_search_or_model_invocation(self):
        def unexpected_search(*args, **kwargs):
            self.fail("invalid view must not search")

        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "view"):
            run({"selected_startup": {"name": "기업"}}, view="technical",
                search=unexpected_search, model=model)
        with self.assertRaisesRegex(ValueError, "view"):
            analyze_business({}, [], view="technical", model=model)
        self.assertEqual(model.calls, 0)

    def test_view_does_not_require_funding_team_or_sufficiency_metadata(self):
        state = {"selected_startup": {"name": "기업"}}
        legacy = build_search_plan(state)
        self.assertIn("입력 누락: selected_startup.team", legacy.information_gaps)
        for view in ("stakeholder", "domain"):
            gaps = build_search_plan(state, view=view).information_gaps
            self.assertIn("입력 누락: selected_startup.product", gaps)
            self.assertNotIn("입력 누락: selected_startup.team", gaps)
            self.assertNotIn("입력 누락: selected_startup.funding_stage", gaps)
            self.assertNotIn("입력 누락: profile.info_sufficiency", gaps)

    def test_empty_search_does_not_fabricate_view_evidence_or_call_model(self):
        for view in ("stakeholder", "domain"):
            with self.subTest(view=view):
                model = FakeModel()
                result = run({"selected_startup": {"name": "기업"}}, view=view,
                             search=lambda *args, **kwargs: [], model=model)
                self.assertEqual(result["business_analysis"]["commercialization_stage"], "unknown")
                self.assertEqual(result["references"], [])
                self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
