import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.contracts import guard_node  # noqa: E402
from ai_investment.models import Evidence  # noqa: E402
from ai_investment.orchestration import (  # noqa: E402
    MAX_REWORK,
    REQUIRED_ITEMS,
    VIEW_NODES,
    VIEWS,
    build_view_result,
    failed_view_result,
    is_sufficient,
    route_from_supervisor,
    supervisor,
)

from ai_investment.state import create_initial_state, dedupe_by_source_id, merge_dict  # noqa: E402


def ev(i, host):
    return Evidence(
        source_id=f"s{i}", title=f"Doc {i}", url=f"https://{host}/a{i}", excerpt="text"
    )


# 서로 다른 사이트 2곳 → 독립 출처 2개
REFS = [ev(1, "news.example.com"), ev(2, "www.paper.org"), ev(3, "news.example.com")]


def full(view, source_ids=("s1", "s2")):
    """필수 항목을 전부 인용으로 채운 결과."""
    claims = [
        {"text": item, "source_ids": list(source_ids), "item": item}
        for item in REQUIRED_ITEMS[view]
    ]
    return build_view_result(view, summary="ok", claims=claims)


def partial(view, drop):
    claims = [
        {"text": item, "source_ids": ["s1", "s2"], "item": item}
        for item in REQUIRED_ITEMS[view]
        if item != drop
    ]
    return build_view_result(view, summary="부족", claims=claims)


def base_state(**overrides):
    state = create_initial_state(
        domain="Physical AI", criteria={}, max_iterations=1, trace_id="t-test"
    )
    state["references"] = list(REFS)
    state["selected_startup"] = {"name": "A"}
    state.update(overrides)
    return state


class ViewResultTests(unittest.TestCase):
    def test_missing_items_are_required_items_without_citation(self):
        claims = [
            {"text": "a", "source_ids": ["s1"], "item": "시장 규모"},
            {"text": "b", "source_ids": [], "item": "수요 근거"},  # 인용 없음 → 미충족
        ]
        result = build_view_result("market", summary="", claims=claims)
        self.assertEqual(result["missing_items"], ["수요 근거", "경쟁 구도"])
        self.assertEqual(result["source_ids"], ["s1"])

    def test_same_site_counts_as_one_source(self):
        result = full("market", source_ids=("s1", "s3"))  # 둘 다 news.example.com
        self.assertFalse(is_sufficient(result, REFS))
        self.assertTrue(is_sufficient(full("market"), REFS))

    def test_failed_result_is_never_sufficient(self):
        self.assertFalse(is_sufficient(failed_view_result("market", "timeout", full("market")), REFS))


class SupervisorRoutingTests(unittest.TestCase):
    def test_first_round_dispatches_all_views_in_parallel(self):
        update = supervisor(base_state())
        self.assertEqual(update["next_views"], list(VIEWS))
        self.assertEqual(set(update["rework_count"].values()), {0})  # 최초 실행은 재작업이 아님
        self.assertEqual(route_from_supervisor({**base_state(), **update}), list(VIEW_NODES.values()))

    def test_only_insufficient_view_is_reworked_with_hints(self):
        analyses = {v: full(v) for v in VIEWS}
        analyses["domain"] = partial("domain", "도입 사례·PoC")
        update = supervisor(base_state(analyses=analyses))
        self.assertEqual(update["next_views"], ["domain"])
        self.assertEqual(update["rework_hints"]["domain"], ["도입 사례·PoC"])
        self.assertEqual(update["rework_hints"]["market"], [])
        self.assertEqual(update["rework_count"]["domain"], 1)

    def test_different_state_selects_different_view(self):
        analyses = {v: full(v) for v in VIEWS}
        analyses["technical"] = partial("technical", "외부 검증")
        update = supervisor(base_state(analyses=analyses))
        self.assertEqual(update["next_views"], ["technical"])

    def test_all_sufficient_goes_to_decision(self):
        state = base_state(analyses={v: full(v) for v in VIEWS})
        update = supervisor(state)
        self.assertEqual(update["next_views"], [])
        self.assertEqual(route_from_supervisor({**state, **update}), "investment_decision")

    def test_rework_limit_stops_retrying(self):
        analyses = {v: full(v) for v in VIEWS}
        analyses["market"] = partial("market", "시장 규모")
        update = supervisor(base_state(
            analyses=analyses, rework_count={v: 0 for v in VIEWS} | {"market": MAX_REWORK}
        ))
        self.assertEqual(update["next_views"], [])

    def test_failed_view_is_retried_within_limit(self):
        analyses = {v: full(v) for v in VIEWS}
        analyses["stakeholder"] = failed_view_result("stakeholder", "APIError")
        update = supervisor(base_state(analyses=analyses))
        self.assertEqual(update["next_views"], ["stakeholder"])

    def test_step_limit_forces_decision(self):
        update = supervisor(base_state(step_count=20, max_steps=20))
        self.assertEqual(update["next_views"], [])
        self.assertEqual(update["step_count"], 21)

    def test_eval_research_request_is_handled_once_then_resumes_report(self):
        verdict = {
            "passed": False,
            "items": {"bias": {"passed": False, "reason": "단일 출처 편중"}},
            "action": "research",
            "target_views": ["market"],
            "feedback": [],
            "research_hints": {"market": ["경쟁사 외부 자료"]},
        }
        state = base_state(
            analyses={v: full(v) for v in VIEWS}, eval_result=verdict,
            report_attempts=1, resume_after_research="report_generation",
        )
        first = supervisor(state)
        self.assertEqual(first["next_views"], ["market"])
        self.assertIsNone(first["eval_result"])            # 요청 소비
        self.assertEqual(first["rework_hints"]["market"], ["경쟁사 외부 자료"])

        # 재조사 후: 같은 요청을 반복하지 않고 보고서 단계로 복귀
        after = {**state, **first}
        second = supervisor(after)
        self.assertEqual(second["next_views"], [])
        self.assertEqual(route_from_supervisor({**after, **second}), "report_generation")

    def test_eval_hints_fall_back_to_failed_item_reasons(self):
        verdict = {
            "passed": False,
            "items": {"bias": {"passed": False, "reason": "단일 출처 편중"}},
            "action": "research", "target_views": ["market"], "feedback": [],
        }
        update = supervisor(base_state(analyses={v: full(v) for v in VIEWS}, eval_result=verdict))
        self.assertIn("단일 출처 편중", update["rework_hints"]["market"][0])

    def test_supervisor_writes_only_control_keys(self):
        guarded = guard_node("supervisor", supervisor)
        self.assertEqual(
            set(guarded(base_state())),
            {"next_views", "rework_count", "rework_hints", "step_count"},
        )


class DecisionLogTests(unittest.TestCase):
    def test_supervisor_logs_decision_and_reasons_with_trace_id(self):
        import importlib
        from unittest import mock

        sup = importlib.import_module("ai_investment.orchestration.supervisor")

        analyses = {v: full(v) for v in VIEWS}
        analyses["domain"] = partial("domain", "운영 제약")
        with mock.patch.object(sup, "log_decision") as log:
            supervisor(base_state(analyses=analyses))

        trace_id, node, decision, reasons = log.call_args.args
        self.assertEqual((trace_id, node, decision), ("t-test", "supervisor", ["domain"]))
        self.assertIn("운영 제약", reasons["domain"])
        self.assertEqual(reasons["technical"], "충분")


class ReducerTests(unittest.TestCase):
    def test_merge_dict_keeps_other_views(self):
        merged = merge_dict({"technical": 1}, {"market": 2})
        self.assertEqual(merged, {"technical": 1, "market": 2})

    def test_references_dedupe(self):
        self.assertEqual(len(dedupe_by_source_id([REFS[0]], [REFS[0], REFS[1]])), 2)


# ---------------------------------------------------------------------------
# langgraph 없이 supervisor ↔ 관점 노드 루프를 reducer 규칙대로 실행해 본다.
# ---------------------------------------------------------------------------

REDUCERS = {"analyses": merge_dict, "node_status": merge_dict, "node_errors": merge_dict,
            "references": dedupe_by_source_id}


def apply(state, update):
    for key, value in update.items():
        state[key] = REDUCERS[key](state.get(key), value) if key in REDUCERS else value


def simulate(view_behaviors, max_steps=20):
    """view_behaviors[view](call_no, hints) -> ViewResult. 방문 경로를 돌려준다."""
    state = base_state(max_steps=max_steps)
    calls = {v: 0 for v in VIEWS}
    path = []
    while True:
        apply(state, supervisor(state))
        nxt = route_from_supervisor(state)
        path.append(nxt)
        if nxt == "investment_decision":
            return path, state
        updates = []
        for node in nxt:  # 병렬 실행: 모두 같은 state를 읽고 결과는 reducer로 합친다
            view = next(v for v, n in VIEW_NODES.items() if n == node)
            hints = state["rework_hints"].get(view, [])
            result = view_behaviors[view](calls[view], hints)
            calls[view] += 1
            updates.append({"analyses": {view: result}, "node_status": {view: result["status"]}})
        for update in updates:
            apply(state, update)


class SimulatedLoopTests(unittest.TestCase):
    def test_reworked_view_recovers_then_proceeds(self):
        def domain(call, hints):
            if call == 0:
                return partial("domain", "도입 사례·PoC")
            assert hints == ["도입 사례·PoC"]  # supervisor 지시가 에이전트에 전달됨
            return full("domain")

        behaviors = {v: (lambda call, hints, v=v: full(v)) for v in VIEWS}
        behaviors["domain"] = domain
        path, state = simulate(behaviors)

        self.assertEqual(path, [list(VIEW_NODES.values()), ["domain"], "investment_decision"])
        self.assertEqual(state["rework_count"], {"technical": 0, "market": 0, "stakeholder": 0, "domain": 1})

    def test_never_sufficient_view_terminates_at_rework_limit(self):
        behaviors = {v: (lambda call, hints, v=v: full(v)) for v in VIEWS}
        behaviors["market"] = lambda call, hints: partial("market", "시장 규모")
        path, state = simulate(behaviors)

        self.assertEqual(path[-1], "investment_decision")
        self.assertEqual(path.count(["market"]), MAX_REWORK)
        self.assertEqual(state["rework_count"]["market"], MAX_REWORK)

    def test_every_view_failing_still_terminates(self):
        behaviors = {v: (lambda call, hints, v=v: failed_view_result(v, "boom")) for v in VIEWS}
        path, _ = simulate(behaviors)
        self.assertEqual(path[-1], "investment_decision")
        self.assertLessEqual(len(path), 1 + MAX_REWORK + 1)


if __name__ == "__main__":
    unittest.main()
