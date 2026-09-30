import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business_search import build_search_plan, collect_evidence
from ai_investment.models import Evidence


class BusinessSearchTests(unittest.TestCase):
    def input(self, suffix):
        path = Path(__file__).parent / "fixtures" / f"business_input_{suffix}.json"
        return json.loads(path.read_text())

    def test_complete_input_uses_product_and_customer_without_mutation(self):
        state = self.input("complete")
        original = deepcopy(state)
        plan = build_search_plan(state)
        self.assertEqual(plan.information_gaps, ())
        self.assertEqual(len(plan.queries), 3)
        self.assertIn(state["selected_startup"]["product"], plan.queries[0])
        self.assertIn(state["profile"]["paying_customer"], plan.queries[0])
        self.assertEqual(state, original)

    def test_name_only_records_gaps_and_searches_company(self):
        plan = build_search_plan(self.input("name_only"))
        self.assertIn("입력 누락: selected_startup.product", plan.information_gaps)
        self.assertIn("입력 누락: profile.paying_customer", plan.information_gaps)
        self.assertTrue(all(query.startswith("가상제조로봇 ") for query in plan.queries))

    def test_missing_name_and_wrong_types_fail_before_search(self):
        for state in ({}, {"selected_startup": {"name": "  "}}):
            with self.assertRaises(ValueError):
                build_search_plan(state)
        for field, value in (("name", 12), ("team", "person"), ("product", [])):
            with self.assertRaises(TypeError):
                build_search_plan({"selected_startup": {"name": "기업", field: value}})

    def test_upstream_gaps_preserved_even_if_sufficiency_claims_enough(self):
        state = self.input("complete")
        state["profile"]["paying_customer"] = " "
        state["profile"]["missing_info"] = ["계약 기간 미확인", "계약 기간 미확인"]
        plan = build_search_plan(state)
        self.assertIn("입력 누락: profile.paying_customer", plan.information_gaps)
        self.assertEqual(plan.information_gaps.count("계약 기간 미확인"), 1)
        self.assertNotIn(state["selected_startup"]["product"], plan.queries[0])

    def test_three_calls_merge_source_excerpts(self):
        calls = []

        def fake_search(query, *, max_results):
            calls.append((query, max_results))
            return [Evidence(source_id="same", title="Source", excerpt=f"근거 {len(calls)}")]

        results = collect_evidence(build_search_plan(self.input("complete")), search=fake_search)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(limit == 5 for _, limit in calls))
        self.assertEqual(len(results), 1)
        self.assertIn("근거 1", results[0].excerpt)
        self.assertIn("근거 3", results[0].excerpt)

    def test_empty_results_and_failure_are_distinct(self):
        plan = build_search_plan(self.input("name_only"))
        self.assertEqual(collect_evidence(plan, search=lambda *args, **kwargs: []), [])

        def failing_search(*args, **kwargs):
            raise RuntimeError("search unavailable")

        with self.assertRaisesRegex(RuntimeError, "search unavailable"):
            collect_evidence(plan, search=failing_search)


if __name__ == "__main__":
    unittest.main()
