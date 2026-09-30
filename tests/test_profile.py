import unittest
from unittest.mock import patch

from ai_investment.agents import profile
from ai_investment.agents.profile import CustomerProfile, classify


class FakeClassifier:
    """Stands in for the structured-output LLM; records what it was sent."""

    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


GOOD = CustomerProfile(
    subdomain="산업용·제조 로봇",
    paying_customer="제조 공장",
    customer_problem="숙련 작업자 부족으로 조립 자동화가 어렵다.",
    info_sufficiency="충분",
)
STARTUP = {
    "name": "카본식스",
    "product": "센서 글러브 시연으로 로봇이 학습하는 시그마키트",
    "funding_stage": "Series A",
    "team": ["문태연"],
}


class ProfileAgentTest(unittest.TestCase):
    def test_classify_returns_plain_dict_with_expected_keys(self):
        fake = FakeClassifier(GOOD)
        result = classify(STARTUP, fake)
        self.assertEqual(result["subdomain"], "산업용·제조 로봇")
        self.assertEqual(
            set(result),
            {
                "subdomain",
                "paying_customer",
                "customer_problem",
                "info_sufficiency",
                "missing_info",
            },
        )
        self.assertIn("카본식스", fake.messages[1][1])

    def test_dict_result_is_validated(self):
        bad = {**GOOD.model_dump(), "subdomain": "우주 탐사"}
        with self.assertRaises(ValueError):
            classify(STARTUP, FakeClassifier(bad))

    def test_missing_product_skips_llm_and_flags_insufficient(self):
        fake = FakeClassifier(GOOD)
        result = classify({"name": "무명로보틱스"}, fake)
        self.assertIsNone(fake.messages)
        self.assertEqual(result["info_sufficiency"], "부족")
        self.assertEqual(result["missing_info"], ["제품 설명"])

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError):
            classify({"product": "x"}, FakeClassifier(GOOD))

    def test_run_writes_only_owned_keys(self):
        with patch.object(profile, "_build_classifier", lambda: FakeClassifier(GOOD)):
            update = profile.run({"selected_startup": STARTUP})
        self.assertEqual(set(update), {"profile", "references"})
        self.assertEqual(update["references"], [])

    def test_run_requires_selected_startup(self):
        with self.assertRaises(ValueError):
            profile.run({})

    def test_guard_accepts_result(self):
        from ai_investment.contracts import guard_node

        with patch.object(profile, "_build_classifier", lambda: FakeClassifier(GOOD)):
            guarded = guard_node("customer_profile", profile.run)
            self.assertIn("profile", guarded({"selected_startup": STARTUP}))


if __name__ == "__main__":
    unittest.main()