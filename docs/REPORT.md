# 보고서 생성 Agent

현재 보고서 구현은 Supervisor v2 공통 계약을 사용합니다.

- `run(state)`: PDF 경로(`report`)와 구조화 초안(`report_draft`)을 함께 반환합니다.
  통과 또는 3번째 미달 후 재호출되면 모델 없이 최종 평가 상태를 출력합니다.
- `finalize_report(state)`: 평가된 본문을 유지하고 최종 `eval_result`의 통과·미통과 사유를 PDF에 반영합니다.
- 입력은 `analyses`, 기존 투자 점수·판정, 현재 대상의 출처, 이전 초안과 `eval_result.feedback`입니다.
- SUMMARY·REFERENCE·4관점·risks·decision_basis·근거 부족 항목을 표시하며 최대 10쪽입니다.
- 자세한 스키마, 그래프 연결, 실행 및 제출 방법은 [Supervisor 보고서 문서](SUPERVISOR_REPORT.md)를 참고하세요.
