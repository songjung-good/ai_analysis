import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.models import Evidence
from ai_investment.orchestration import evaluator as ev
from ai_investment.orchestration.evaluator import JudgeItem, JudgeOutput


def ref(sid, url):
    return Evidence(source_id=sid, title=f"자료 {sid}", url=url, excerpt=f"발췌 {sid}")


REFS = [
    ref("S1", "https://ifr.org/a"), ref("S2", "https://reuters.com/b"),
    ref("S3", "https://etnews.com/c"), ref("S4", "https://kiat.or.kr/d"),
    ref("S5", "https://bloomberg.com/e"), ref("S6", "https://mk.co.kr/f"),
]


def para(text, *sids):
    return {"text": text, "source_ids": list(sids)}


def good_draft():
    return {
        "summary": [para("요약", "S1")],
        "technical": [para("기술 근거", "S1"), para("기술 한계", "S2")],
        "market": [para("시장 규모", "S3"), para("수요", "S4")],
        "stakeholder": [para("고객", "S5"), para("공급사", "S6")],
        "domain": [para("적용 현장", "S2"), para("PoC", "S4")],
        "risks": [para("인증 리스크", "S5")],
        "decision_basis": [para("가중합 3.4점으로 conditional", "S3")],
    }


def judge_all(passed=True, **overrides):
    items = {k: JudgeItem(passed=passed, reason="ok" if passed else "문제") for k in ev.ITEMS}
    items.update(overrides)
    return lambda prompt: JudgeOutput(**items)


class Calls:
    def __init__(self, fn):
        self.fn, self.n, self.prompts = fn, 0, []

    def __call__(self, prompt):
        self.n += 1
        self.prompts.append(prompt)
        return self.fn(prompt)


def state(draft, attempts=0):
    return {"report_draft": draft, "references": REFS, "report_attempts": attempts, "trace_id": "t-1"}


class RuleTests(unittest.TestCase):
    def test_good_draft_passes_all_rules(self):
        rules = ev.check_rules(good_draft(), ev._refs_by_id(REFS))
        self.assertTrue(all(r["passed"] and not r["warning"] for r in rules.values()), rules)

    def test_uncited_paragraphs_fail_groundedness(self):
        d = good_draft()
        d["market"] = [para("근거 없는 시장 주장"), para("또 근거 없음")]
        r = ev.check_groundedness(d, ev._refs_by_id(REFS))
        self.assertFalse(r["passed"])
        self.assertEqual(r["views"], ["market"])

    def test_unknown_source_fails_groundedness(self):
        d = good_draft()
        d["technical"][0]["source_ids"] = ["S99"]
        self.assertIn("S99", ev.check_groundedness(d, ev._refs_by_id(REFS))["reason"])

    def test_banned_phrase_is_only_a_warning(self):
        d = good_draft()
        d["summary"] = [para("압도적인 기술력", "S1")]
        r = ev.check_neutrality(d)
        self.assertTrue(r["passed"])
        self.assertTrue(r["warning"])

    def test_single_domain_view_fails_bias(self):
        d = good_draft()
        d["stakeholder"] = [para("고객", "S5"), para("공급사", "S5")]
        r = ev.check_bias(d, ev._refs_by_id(REFS))
        self.assertFalse(r["passed"])
        self.assertEqual(r["views"], ["stakeholder"])

    def test_missing_risk_section_fails_bias(self):
        d = good_draft()
        d["risks"] = []
        self.assertFalse(ev.check_bias(d, ev._refs_by_id(REFS))["passed"])

    def test_empty_view_fails_coverage(self):
        d = good_draft()
        d["domain"] = []
        r = ev.check_coverage(d)
        self.assertFalse(r["passed"])
        self.assertEqual(r["views"], ["domain"])


class EvaluateTests(unittest.TestCase):
    def test_pass(self):
        judge = Calls(judge_all(True))
        out = ev.evaluate(state(good_draft()), judge=judge)
        self.assertTrue(out["eval_result"]["passed"])
        self.assertEqual(out["eval_result"]["action"], "pass")
        self.assertEqual(out["report_attempts"], 0)
        self.assertEqual(judge.n, 1)
        self.assertIn("발췌 S1", judge.prompts[0])  # Judge는 실제 근거 발췌를 본다
        self.assertEqual(ev.route_after_eval({**state(good_draft()), **out}), "end")

    def test_rule_hard_fail_skips_judge_and_requests_research(self):
        d = good_draft()
        d["domain"] = []
        judge = Calls(judge_all(True))
        out = ev.evaluate(state(d), judge=judge)
        v = out["eval_result"]
        self.assertEqual(judge.n, 0)
        self.assertFalse(v["passed"])
        self.assertEqual(v["action"], "research")
        self.assertIn("domain", v["target_views"])
        self.assertEqual(out["report_attempts"], 1)
        self.assertEqual(ev.route_after_eval({**state(d), **out}), "supervisor")

    def test_judge_style_problem_requests_rewrite(self):
        bad = JudgeItem(passed=False, reason="단정적 표현", fix=["'확실히 성장' 표현을 수치 근거로 바꿀 것"])
        out = ev.evaluate(state(good_draft()), judge=judge_all(True, neutrality=bad))
        v = out["eval_result"]
        self.assertEqual(v["action"], "rewrite")
        self.assertTrue(any("확실히 성장" in f for f in v["feedback"]))
        self.assertEqual(ev.route_after_eval({**state(good_draft()), **out}), "report_generation")

    def test_judge_evidence_problem_requests_research(self):
        bad = JudgeItem(passed=False, reason="발췌가 주장 미지지", problem_views=["technical", "bogus"])
        v = ev.evaluate(state(good_draft()), judge=judge_all(True, groundedness=bad))["eval_result"]
        self.assertEqual(v["action"], "research")
        self.assertEqual(v["target_views"], ["technical"])

    def test_judge_failure_is_not_a_pass(self):
        def broken(prompt):
            raise ValueError("parse error")
        judge = Calls(broken)
        out = ev.evaluate(state(good_draft()), judge=judge)
        self.assertEqual(judge.n, 2)  # 1회 재호출
        self.assertFalse(out["eval_result"]["passed"])
        self.assertEqual(out["eval_result"]["action"], "rewrite")

    def test_loop_ends_at_attempt_limit(self):
        d = good_draft()
        d["domain"] = []
        out = ev.evaluate(state(d, attempts=ev.MAX_REPORT_ATTEMPTS - 1), judge=judge_all(True))
        self.assertEqual(out["report_attempts"], ev.MAX_REPORT_ATTEMPTS)
        self.assertEqual(ev.route_after_eval({**state(d), **out}), "end")


if __name__ == "__main__":
    unittest.main()
