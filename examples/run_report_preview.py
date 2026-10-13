"""Offline v2 layout preview with synthetic data. Not a judged report or trace."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_investment.agents.report import ReportDraft, ReportParagraph, _prepare_draft, _used_references, render_report_pdf
from ai_investment.models import Evidence
from ai_investment.state import VIEWS, create_initial_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/supervisor-report-sample.pdf"))
    args = parser.parse_args()
    state = create_initial_state(domain="Physical AI / Robotics", criteria={}, max_iterations=1, trace_id="sample-layout-v2")
    state.update({
        "selected_startup": {"name": "가상로보틱스 (레이아웃 검증용)"},
        "analyses": {view: {"view": view, "summary": "가상 샘플", "claims": [], "source_ids": ["sample-product"],
                            "missing_items": ["독립적인 실제 자료 필요"], "status": "done"} for view in VIEWS},
        "references": [Evidence(source_id="sample-product", title="가상 제품 설명 - 테스트용 자료",
                                url="https://example.com/sample", page=1,
                                excerpt="이 자료와 기업은 테스트를 위해 만든 가상 데이터입니다.")],
    })
    contents = {
        "summary": "이 문서는 네 관점 보고서의 한글과 인용 배치를 확인하기 위한 가상 샘플입니다. 실제 기업 평가 결과가 아닙니다.",
        "company": "가상 로봇 기업이며 실제 기업의 투자 검토에 사용할 수 없습니다.",
        "technical": "기술 성능을 뒷받침하는 실제 자료는 제공되지 않았습니다. 성능 수치를 임의로 추정하지 않습니다.",
        "market": "시장 규모와 비용 자료는 정보 부족입니다. 시장 범위와 기준 시점을 함께 확인해야 합니다.",
        "stakeholder": "구매자, 운영자, 최종 사용자의 요구사항은 확인되지 않았습니다. 이해관계자별 이점과 부담을 추가 조사해야 합니다.",
        "domain": "현장 적용 조건은 정보 부족입니다. 작업 환경과 운영 제약을 검증해야 합니다.",
        "risks": "자료 부족과 실제 위험 발생은 구분해야 합니다. 독립적인 출처가 없어 편향과 성능 불확실성이 남아 있습니다.",
        "decision_basis": "실제 점수와 투자 판단을 산정하지 않은 샘플입니다. 코드가 계산한 판단을 실제 보고서에서 그대로 표시해야 합니다.",
    }
    draft = ReportDraft(**{key: [ReportParagraph(text=text, source_ids=["sample-product"] if key == "summary" else [])]
                           for key, text in contents.items()})
    draft = _prepare_draft(draft, state)
    print(render_report_pdf(draft, state, _used_references(draft, state), args.output, sample=True))


if __name__ == "__main__":
    main()
