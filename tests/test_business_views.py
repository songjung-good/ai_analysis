import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.agents.business import VIEW_REQUIREMENTS, run, to_view_result
from ai_investment.agents.business_analysis import analyze_business
from ai_investment.agents.business_models import BusinessAnalysis
from ai_investment.agents.business_search import build_search_plan
from ai_investment.contracts import guard_node
from ai_investment.models import Evidence
from ai_investment.state import GraphState, create_initial_state


class FakeModel:
    def __init__(self, output=None):
        self.messages = None
        self.calls = 0
        self.output = output

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        self.calls += 1
        if self.output is not None:
            return deepcopy(self.output)
        view = json.loads(messages[1][1]).get("view")
        return {
            "commercialization_stage": "unknown", "stage_reason": "현장 단계 미확인",
            "stage_source_ids": [], "customer_cases": [], "deployment_costs": [],
            "deployment_effects": [], "regulatory_risks": [],
            "evidence_assessment": [{
                "claim": "고객이 설치 비용을 부담한다.", "level": "single_source",
                "reason": "고객 자료", "source_ids": ["customer"],
            }],
            "information_gaps": [], "summary": "고객의 설치 비용 부담이 확인됐다.",
            "view_findings": [{
                "topic": "cost_burden" if view == "stakeholder" else "constraints",
                "text": "고객이 설치 비용을 부담한다.", "source_ids": ["customer"],
            }] if view else [],
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
                self.assertEqual(set(result), {"analyses", "references", "node_status", "node_errors"})
                self.assertEqual(set(result["analyses"]), {view})
                self.assertEqual(result["node_status"], {view: "done"})
                self.assertEqual(result["node_errors"], {view: None})
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
                analysis = result["analyses"][view]
                self.assertEqual(analysis["status"], "done")
                self.assertEqual(analysis["claims"], [])
                self.assertTrue(analysis["missing_items"])
                self.assertEqual(result["references"], [])
                self.assertEqual(model.calls, 0)

    def complete_result(self, view):
        return BusinessAnalysis(
            commercialization_stage="unknown", stage_reason="운영 단계 미확인",
            stage_source_ids=[], customer_cases=[], deployment_costs=[], deployment_effects=[],
            regulatory_risks=[], evidence_assessment=[], summary="관점별 확인 내용",
            information_gaps=["입력 누락: selected_startup.product", "다른 관점 조사 이력"],
            view_findings=[{
                "topic": topic, "text": f"{topic} 관련 확인 내용",
                "source_ids": ["customer", "supplier"],
            } for topic in VIEW_REQUIREMENTS[view]],
        )

    def test_adapter_requires_cited_topic_coverage_and_preserves_view_specific_gaps(self):
        for view in ("stakeholder", "domain"):
            with self.subTest(view=view):
                raw = self.complete_result(view)
                before = raw.model_dump()
                converted = to_view_result(raw, view=view)
                self.assertEqual(converted["missing_items"], [])
                self.assertEqual(converted["source_ids"], ["customer", "supplier"])
                self.assertEqual(raw.model_dump(), before)
                raw = raw.model_copy(update={
                    "view_findings": raw.view_findings[1:],
                    "view_missing_items": ["현재 계약 범위 상충"],
                })
                converted = to_view_result(raw, view=view)
                self.assertIn("현재 계약 범위 상충", converted["missing_items"])
                self.assertIn(next(iter(VIEW_REQUIREMENTS[view].values())), converted["missing_items"])

    def test_adapter_rejects_other_view_topics(self):
        with self.assertRaisesRegex(ValueError, "another view"):
            to_view_result(self.complete_result("domain"), view="stakeholder")

    def test_findings_with_unknown_citations_return_failed_view(self):
        raw = self.complete_result("stakeholder").model_dump()
        raw["view_findings"][0]["source_ids"] = ["invented"]
        result = guard_node("stakeholder", lambda state: run(
            state, view="stakeholder", model=FakeModel(raw),
            search=lambda *args, **kwargs: [
                Evidence(source_id="customer", title="고객", excerpt="고객 자료"),
                Evidence(source_id="supplier", title="공급사", excerpt="공급사 자료"),
            ],
        ))({"selected_startup": {"name": "기업"}})
        self.assertEqual(result["analyses"]["stakeholder"]["status"], "failed")
        self.assertIn("Unknown evidence source IDs", result["node_errors"]["stakeholder"])
        self.assertEqual(result["references"], [])

    def test_two_business_views_merge_without_shared_legacy_field_or_state_mutation(self):
        builder = StateGraph(GraphState)
        state = create_initial_state(domain="Physical AI", criteria={}, max_iterations=1)
        state.update(selected_startup={"name": "기업"}, business_analysis={"summary": "기존 분석"})
        before = deepcopy(state)
        sources = [
            Evidence(source_id="customer", title="고객", excerpt="고객 자료"),
            Evidence(source_id="supplier", title="공급사", excerpt="공급사 자료"),
        ]
        for view in ("stakeholder", "domain"):
            def worker(value, view=view):
                return run(value, view=view, model=FakeModel(self.complete_result(view).model_dump()),
                           search=lambda *args, **kwargs: sources)

            builder.add_node(view, guard_node(view, worker))
            builder.add_edge(START, view)
            builder.add_edge(view, END)
        result = builder.compile().invoke(state)
        self.assertEqual(set(result["analyses"]), {"stakeholder", "domain"})
        self.assertEqual(result["node_errors"], {"stakeholder": None, "domain": None})
        self.assertEqual(result["business_analysis"], {"summary": "기존 분석"})
        self.assertEqual(result["references"], sources)
        self.assertEqual(state, before)

    def test_search_failure_returns_only_the_failed_view_and_legacy_call_still_raises(self):
        def failing_search(*args, **kwargs):
            raise RuntimeError("search unavailable")

        state = {"selected_startup": {"name": "기업"}}
        result = run(state, view="domain", search=failing_search, model=FakeModel())
        self.assertEqual(result["node_status"], {"domain": "failed"})
        self.assertEqual(result["node_errors"], {"domain": "RuntimeError: search unavailable"})
        self.assertEqual(set(result["analyses"]), {"domain"})
        with self.assertRaisesRegex(RuntimeError, "search unavailable"):
            run(state, search=failing_search, model=FakeModel())

    def test_rework_reuses_only_own_cited_evidence_and_logs_changed_queries(self):
        state = create_initial_state(domain="Physical AI", criteria={}, max_iterations=1, trace_id="rework-test")
        state["selected_startup"] = {"name": "기업"}
        original = self.complete_result("domain")
        initial_raw = original.model_copy(update={"view_findings": original.view_findings[:-1]})
        prior_sources = [
            Evidence(source_id="customer", title="고객", excerpt="기업 현장 적용 사례"),
            Evidence(source_id="supplier", title="공급사", excerpt="기업 PoC 단계"),
        ]
        first_queries = []

        def first_search(query, **kwargs):
            first_queries.append(query)
            return prior_sources

        first = run(state, view="domain", search=first_search, model=FakeModel(initial_raw.model_dump()))
        state.update(first)
        hint = "안전 정지 조건과 운영 제약 확인"
        state["rework_hints"] = {"domain": [hint], "stakeholder": ["다른 관점 비용 부담"]}
        state["rework_count"]["domain"] = 1
        state["references"].append(Evidence(source_id="other", title="다른 관점", excerpt="다른 관점 자료"))
        before = deepcopy(state)
        latest_raw = original.model_dump()
        latest_raw["view_findings"][-1]["source_ids"] = ["new"]
        latest_raw["view_findings"][-1]["text"] = "비상 정지와 현장 통합 조건이 있다."
        model = FakeModel(latest_raw)
        next_queries = []

        def rework_search(query, **kwargs):
            next_queries.append(query)
            return [Evidence(source_id="new", title="운영 조건", excerpt="비상 정지와 현장 통합 조건")]

        with self.assertLogs("ai_investment.agents.business_search", level="INFO") as logs:
            latest = run(state, view="domain", search=rework_search, model=model)
        payload = json.loads(model.messages[1][1])
        self.assertEqual(payload["rework_hints"], [hint])
        self.assertEqual(payload["previous_result"], first["analyses"]["domain"])
        self.assertEqual({item["source_id"] for item in payload["evidence"]}, {"customer", "supplier", "new"})
        self.assertNotEqual(first_queries, next_queries)
        self.assertEqual(len(next_queries), 3)
        self.assertTrue(all(hint in query and "다른 관점 비용 부담" not in query for query in next_queries))
        started = [record for record in logs.records if record.event == "search_started"]
        self.assertEqual([record.query for record in started], next_queries)
        self.assertTrue(all(record.trace_id == "rework-test" and record.attempt == 1 for record in started))
        self.assertTrue(all(record.target_name == "기업" and record.view == "domain" for record in started))
        self.assertTrue(first["analyses"]["domain"]["missing_items"])
        self.assertEqual(latest["analyses"]["domain"]["missing_items"], [])
        self.assertEqual(latest["analyses"]["domain"]["source_ids"], ["customer", "new", "supplier"])
        self.assertEqual(state, before)

    def test_rework_does_not_clear_unresolved_hint_when_search_is_empty(self):
        state = {
            "selected_startup": {"name": "기업"},
            "rework_hints": {"domain": ["운영 시간 확인"]},
        }
        model = FakeModel()
        result = run(state, view="domain", search=lambda *args, **kwargs: [], model=model)
        self.assertIn("운영 시간 확인", result["analyses"]["domain"]["missing_items"])
        self.assertEqual(result["analyses"]["domain"]["source_ids"], [])
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
