import tempfile
import unittest
from pathlib import Path

import pymupdf

from ai_investment.agents.report import ReportAgent, ReportDraft, ReportParagraph
from ai_investment.contracts import guard_node
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.models import Evidence
from ai_investment.state import create_initial_state


def sample_state():
    state = create_initial_state(
        domain="Physical AI/Robotics", criteria={}, max_iterations=2,
    )
    state.update({
        "candidates": [{"name": "가상로보틱스", "product": "물류 로봇", "funding_stage": "Seed", "team": ["가상 창업자"]}],
        "current_idx": 0,
        "selected_startup": {"name": "가상로보틱스", "product": "물류 로봇", "funding_stage": "Seed", "team": ["가상 창업자"]},
        "profile": {"subdomain": "물류 로봇", "paying_customer": "물류센터"},
        "scores": {"team": 3, "market": 3, "technology": 4, "competition": 3, "traction": 2, "investment_terms": 3},
        "investment_score": 3.05,
        "decision": "conditional",
        "decision_reason": "고객 계약 검증이 필요합니다.",
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
        technology_market=[ReportParagraph(text="기술 성능 자료는 정보 부족입니다.")],
        evaluation=[ReportParagraph(text="투자 조건 확인이 필요합니다.")],
        recommendation=[ReportParagraph(text="고객 계약을 후속 확인합니다.")],
    )


class ReportAgentTests(unittest.TestCase):
    def test_five_page_pdf_has_cited_references_and_preserves_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            state = sample_state()
            before = dict(state)
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _state: sample_draft())
            update = guard_node("report_generation", agent.run)(state)
            self.assertEqual(set(update), {"report"})
            self.assertEqual(state, before)
            path = Path(update["report"])
            self.assertTrue(path.is_absolute() and path.is_file())
            with pymupdf.open(path) as pdf:
                self.assertEqual(pdf.page_count, 5)
                self.assertIn("SUMMARY", pdf[0].get_text())
                self.assertIn("가상로보틱스", pdf[0].get_text())
                self.assertIn("3.05 / 5.00", pdf[0].get_text())
                self.assertIn("물류 로봇", pdf[1].get_text())
                self.assertIn("투자 평가 및 최종 의견", pdf[3].get_text())
                self.assertIn("REFERENCE", pdf[-1].get_text())
                self.assertIn("https://example.com/product", pdf[-1].get_text())
                self.assertNotIn("사용하지 않은 자료", pdf[-1].get_text())

    def test_empty_discovery_creates_two_page_pdf_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(
                output_dir=Path(directory),
                draft_writer=lambda _state: self.fail("No candidate must not invoke the LLM"),
            )
            initial = create_initial_state(domain="Physical AI/Robotics", criteria={}, max_iterations=2)
            initial.update(candidates=[], current_idx=0, selected_startup={})
            result = agent.run(initial)
            with pymupdf.open(result["report"]) as pdf:
                self.assertEqual(pdf.page_count, 2)
                self.assertIn("SUMMARY", pdf[0].get_text())
                self.assertIn("후보가 없어", pdf[0].get_text())
                self.assertIn("REFERENCE", pdf[1].get_text())

    def test_unknown_citation_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = sample_draft()
            draft.summary[0].source_ids = ["invented"]
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _state: draft)
            with self.assertRaisesRegex(ValueError, "unknown source_ids"):
                agent.run(sample_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_empty_draft_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _state: ReportDraft())
            with self.assertRaisesRegex(ValueError, "requires summary and recommendation"):
                agent.run(sample_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_uncited_report_with_available_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = sample_draft()
            for section in draft.sections():
                for paragraph in section:
                    paragraph.source_ids = []
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _state: draft)
            with self.assertRaisesRegex(ValueError, "did not cite available evidence"):
                agent.run(sample_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_reference_overflow_leaves_no_partial_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            state = sample_state()
            identifiers = [f"source-{number}" for number in range(25)]
            state["references"] = [
                Evidence(source_id=source_id, title="긴 참고문헌 " * 8 + str(number),
                         url="https://example.com/" + "very-long-address/" * 6 + str(number),
                         excerpt="근거")
                for number, source_id in enumerate(identifiers)
            ]
            draft = ReportDraft(
                summary=[ReportParagraph(text="요약", source_ids=identifiers[:5])],
                company=[ReportParagraph(text="기업", source_ids=identifiers[5:10])],
                technology_market=[ReportParagraph(text="시장", source_ids=identifiers[10:15])],
                evaluation=[ReportParagraph(text="평가", source_ids=identifiers[15:20])],
                recommendation=[ReportParagraph(text="의견", source_ids=identifiers[20:])],
            )
            agent = ReportAgent(output_dir=Path(directory), draft_writer=lambda _state: draft)
            with self.assertRaisesRegex(ValueError, "exceeds the five-page layout"):
                agent.run(state)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_graph_routes_empty_discovery_to_real_report_node(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = ReportAgent(output_dir=Path(directory))

            def unexpected(_state):
                self.fail("Analysis nodes must not run without a candidate")

            graph = build_graph(AgentNodes(
                lambda _state: {"candidates": [], "current_idx": 0, "selected_startup": {}, "references": []},
                unexpected, unexpected, unexpected, unexpected, unexpected, agent.run,
            ))
            result = graph.invoke(create_initial_state(domain="Physical AI/Robotics", criteria={}, max_iterations=1))
            self.assertTrue(Path(result["report"]).is_file())


if __name__ == "__main__":
    unittest.main()
