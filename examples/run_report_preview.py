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
    parser.add_argument("--output", type=Path, default=Path("output/pdf/supervisor-report-layout-v3.pdf"))
    args = parser.parse_args()
    state = create_initial_state(domain="Physical AI / Robotics", criteria={}, max_iterations=1, trace_id="sample-layout-v3")
    state.update({
        "selected_startup": {"name": "가상로보틱스", "product": "물류 로봇", "funding_stage": "Seed"},
        "scores": {"team": 3, "market": 3, "technology": 4, "competition": 3, "traction": 2, "investment_terms": 3},
        "investment_score": 3.05, "decision": "hold",
        "analyses": {view: {"view": view, "summary": "가상 샘플", "claims": [], "source_ids": ["sample-product"],
                            "missing_items": ["독립적인 실제 자료 필요"], "status": "done"} for view in VIEWS},
        "references": [Evidence(source_id="sample-product", title="가상 제품 설명 - 테스트용 자료",
                                url="https://example.com/sample", page=1,
                                excerpt="이 자료와 기업은 테스트를 위해 만든 가상 데이터입니다.")],
    })
    contents = {
        "summary": ["이 문서는 가상 기업과 가상 점수로 구성한 양식 샘플입니다. 실제 기업 평가나 투자 추천에 사용할 수 없습니다.",
                    "물류 로봇 기업의 투자 검토에서 제품 성능, 고객 운영, 비용 조건을 구분해 읽도록 구성했습니다. 미확인 조건은 보류 사유와 후속 조사로 연결합니다."],
        "company": ["샘플의 사업 대상은 물류센터의 반복 이송 업무입니다. 제품·고객·사업 단계에 관한 실제 외부 검증 자료는 제공하지 않았습니다."],
        "technical": ["핵심 판단: 로봇의 기능 설명과 실제 현장 성능은 구분해 검토해야 합니다. 설계 설명만으로 자율주행의 안정성과 운영 효율을 확정할 수 없습니다.",
                      "확인 근거: 이 샘플에는 가상 제품 설명만 있습니다. 실제 보고서에서는 성능 시험의 조건, 제품 세대와 고객 환경을 함께 인용합니다.",
                      "한계: 작업 환경별 오차, 장애물 대응, 연속 운영 시간 등 실제 검증 지표가 필요합니다."],
        "market": ["핵심 판단: 시장 전체의 성장과 해당 기업의 매출 확보 가능성은 별도로 판단해야 합니다.",
                   "확인 근거: 시장 규모 자료는 범위·지역·기준 연도가 일치할 때만 비교합니다. 이 샘플에는 실제 시장 수치가 없습니다.",
                   "한계: 목표 고객 규모, 대체 솔루션과 구매 전환율을 확인해야 합니다."],
        "stakeholder": ["핵심 판단: 구매자, 운영자, 최종 사용자의 이점과 부담을 나눠 살펴야 합니다.",
                        "확인 근거: 이 샘플에는 실제 계약서나 사용자 인터뷰가 없습니다. 고객 이름이나 로고만으로 구매 관계를 확정하지 않습니다.",
                        "한계: 구매 결정권, 설치·운영비 부담, 현장 작업자와의 역할 조정이 미확인입니다."],
        "domain": ["핵심 판단: 현장 적용 가능성은 실제 운영 조건과 함께 검토해야 합니다.",
                   "확인 근거: 고객 현장 시험과 유료 운영을 구분하고, 실증 조건과 운영 결과가 확인된 자료를 사용해야 합니다.",
                   "한계: 통로·바닥 조건, 기존 시스템 통합, 유지보수와 안전 관리가 미확인입니다."],
        "risks": ["고객 계약과 반복 매출을 확인해야 합니다. 제품 소개나 협약만으로 실제 유료 운영을 판단할 수 없습니다.",
                  "독립적인 성능 검증 자료가 필요합니다. 미확인 성능을 실제 결함 발생으로 해석하지 않습니다.",
                  "설치·운영 비용을 확인해야 합니다. 계약 범위와 측정 조건이 없으면 ROI를 계산하지 않습니다."],
        "decision_basis": ["표의 점수는 레이아웃 확인용 가상 값입니다. 실제 실행에서는 코드가 산정한 점수·판단을 그대로 표시합니다.",
                           "재검토에는 고객의 유료 운영 근거, 조건이 명시된 성능 검증, 전체 도입·운영 비용 자료가 필요합니다."],
    }
    draft = ReportDraft(**{key: [ReportParagraph(text=text, source_ids=["sample-product"] if key == "summary" else [])
                               for text in items] for key, items in contents.items()})
    draft = _prepare_draft(draft, state)
    print(render_report_pdf(draft, state, _used_references(draft, state), args.output, sample=True))


if __name__ == "__main__":
    main()
