"""app.py가 supervisor 구조에서 관점 노드를 올바른 진입점으로 연결하는지 확인한다 (API 호출 없음)."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_app():
    spec = importlib.util.spec_from_file_location("app_entry", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AppWiringTests(unittest.TestCase):
    def test_all_supervisor_nodes_are_found(self):
        loaded, missing = _load_app().load_supervisor_nodes()
        self.assertEqual(missing, [])
        self.assertEqual(set(loaded), {"supervisor", "evaluator", "stakeholder", "domain"})

    def test_market_uses_view_entrypoint_in_supervisor_mode(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("run_view as market_view", source)
        self.assertIn("market_view if use_supervisor else market_analysis", source)


if __name__ == "__main__":
    unittest.main()
