import copy
import tempfile
import unittest
from pathlib import Path

import pymupdf

from ai_investment.agents.report import ReportAgent, ReportDraft, ReportParagraph
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.state import create_initial_state, VIEWS


def sample_state():
    state = create_initial_state(domain="Physical AI/Robotics", criteria={}, max_iterations=2, trace_id="report-test-run")
    state.update({
        "candidates": [{"name": "가상로보틱스"}], "current_idx": 0,
        "selected_startup": {"name": "가상로보틱스", "product": "물류 로봇", "funding_stage": "Seed", "team": ["가상 창업자"]},
        "profile": {"subdomain": "물류 로봇", "paying_customer": "물류센터", "source_ids": ["used"]},
        "analyses": {view: {"view": view, "summary": "검증 필요", "claims": [{"text": "물류 로봇 정보", "source_ids": ["used"]}],
                            "source_ids": ["used"], "missing_items": ["독립적인 근거 확인"], "status": "done"} for view in VIEWS},
        "scores": {"team": 3, "market": 3, "technology": 4, "competition": 3, "traction": 2, "investment_terms": 3},
        "investment_score": 3.05, "decision": "conditional", "decision_reason": "고객 계약 검증이 필요합니다.",
        "evaluations": [{"startup_name": "가상로보틱스", "score": 3.05, "decision": "conditional", "reason": "계약 검증 필요"}],
        "references": [
            Evidence(source_id="used", title="가상 기업 제품 설명", url="https://example.com/product", excerpt="물류 로봇 정보"),
            Evidence(source_id="unused", title="사용하지 않은 자료", url="https://example.com/unused", excerpt="다른 정보"),
        ],
    })
    return state


def sample_draft():
    return ReportDraft(
        summary=[ReportParagraph(text="물류 로봇을 개발합니다.", source_ids=["used"])],
        company=[ReportParagraph(text="제품 설명은 출처를 확인했습니다.", source_ids=["used"])],
        **{view: [ReportParagraph(text=f"{view}: 물류 로봇의 적용 근거는 추가 검증이 필요합니다.", source_ids=["used"])] for view in VIEWS},
        risks=[ReportParagraph(text="고객 계약과 성능 검증이 필요합니다.", source_ids=["used"])],
        decision_basis=[ReportParagraph(text="코드 점수에 따른 조건부 검토이며, 고객 계약을 후속 확인합니다.", source_ids=["used"])],
    )


class ReportAgentTests(unittest.TestCase):
    def test_pdf_and_draft_have_citations_and_preserve_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            state = sample_state()
            before = copy.deepcopy(state)
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: sample_draft())
            update = guard_node("report_generation", agent.run)(state)
            self.assertEqual(set(update), {"report", "report_draft"})
            self.assertEqual(state, before)
            path = Path(update["report"])
            self.assertTrue(path.is_absolute() and path.is_file())
            self.assertIn(state["trace_id"], path.name)
            with pymupdf.open(path) as pdf:
                self.assertLessEqual(pdf.page_count, 10)
                text = "\n".join(page.get_text() for page in pdf)
                for expected in ("SUMMARY", "가상로보틱스", "3.05 / 5.00", "REFERENCE", "RISKS", "물류 로봇",
                                 "이해관계자 관점", "도메인 적용 관점", "https://example.com/product"):
                    self.assertIn(expected, text)
                self.assertNotIn("사용하지 않은 자료", text)

    def test_empty_discovery_does_not_call_model_and_stores_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: self.fail("No candidate must not call LLM"))
            initial = create_initial_state(domain="Physical AI/Robotics", criteria={}, max_iterations=2)
            initial.update(candidates=[], current_idx=0, selected_startup={})
            result = agent.run(initial)
            self.assertTrue(result["report_draft"]["summary"])
            with pymupdf.open(result["report"]) as pdf:
                self.assertLessEqual(pdf.page_count, 10)
                text = "\n".join(page.get_text() for page in pdf)
                self.assertIn("후보가 없어", text)
                self.assertIn("REFERENCE", text)

    def test_unknown_citation_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = sample_draft()
            draft.summary[0].source_ids = ["invented"]
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: draft)
            with self.assertRaisesRegex(ValueError, "unknown source_ids"):
                agent.run(sample_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_empty_draft_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: ReportDraft())
            with self.assertRaisesRegex(ValueError, "requires sections"):
                agent.run(sample_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_uncited_information_gaps_remain_visible_for_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = sample_draft()
            for section in draft.sections():
                for paragraph in section:
                    paragraph.source_ids = []
                    paragraph.text = "현재 근거가 부족해 후속 확인이 필요합니다."
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: draft)
            result = agent.run(sample_state())
            self.assertTrue(Path(result["report"]).is_file())
            self.assertTrue(result["report_draft"]["missing_information"])

    def test_reference_overflow_leaves_no_partial_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            state = sample_state()
            ids = [f"source-{i}" for i in range(30)]
            state["references"] = [Evidence(source_id=sid, title="아주 긴 참고문헌 제목 " * 90,
                                            url="https://example.com/" + "long/" * 40, excerpt="근거") for sid in ids]
            state["profile"]["source_ids"] = ids
            for view in VIEWS:
                state["analyses"][view]["source_ids"] = ids
            draft = sample_draft()
            for i, key in enumerate(("summary", *VIEWS, "risks", "decision_basis")):
                getattr(draft, key)[0].source_ids = ids[i*5:(i+1)*5]
            draft.company[0].source_ids = []
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _: draft)
            with self.assertRaisesRegex(ValueError, "ten-page"):
                agent.run(state)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_graph_routes_empty_discovery_to_real_report_node(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(output_dir=Path(directory))
            def unexpected(_state):
                self.fail("Analysis nodes must not run without a candidate")
            graph = build_graph(AgentNodes(
                lambda _: {"candidates": [], "current_idx": 0, "selected_startup": {}, "references": []},
                unexpected, unexpected, unexpected, unexpected, unexpected, agent.run,
            ))
            result = graph.invoke(create_initial_state(domain="Physical AI/Robotics", criteria={}, max_iterations=1))
            self.assertTrue(Path(result["report"]).is_file())
            self.assertTrue(result["report_draft"])


if __name__ == "__main__":
    unittest.main()
