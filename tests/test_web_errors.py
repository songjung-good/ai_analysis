import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.tools.web import _to_evidence


class WebErrorTests(unittest.TestCase):
    def test_provider_error_is_not_an_empty_success(self):
        with self.assertRaises(ConnectionError):
            _to_evidence({"error": ConnectionError("connection unavailable")}, 5)
        with self.assertRaisesRegex(RuntimeError, "Tavily search returned an error"):
            _to_evidence({"error": "provider failure"}, 5)


if __name__ == "__main__":
    unittest.main()
