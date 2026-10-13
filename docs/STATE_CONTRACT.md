# Supervisor 공통 State·쓰기 계약

현재 구현 범위는 공통 State, Reducer, 노드 쓰기 권한이다. 실제 Supervisor·Evaluator·동적 그래프는 각 담당자가 연결한다.

## 공통 타입

`ai_investment.state`에서 `VIEWS`, `ViewResult`, `EvalVerdict`, `GraphState`를 import한다.

- 관점 ID: `technical`, `market`, `stakeholder`, `domain`.
- `ViewResult`: view, summary, claims, missing_items, source_ids, status.
- claim: text, source_ids. 결과 status는 `done` 또는 `failed`.
- `EvalVerdict`: passed, items, action, target_views, feedback. research일 때는 관점별 보완 지시인 `research_hints`도 제공한다.
- 평가 항목: passed, reason. action은 `pass`, `rewrite`, `research`.
- TypedDict는 런타임 검증기가 아니다. 각 담당 노드에서 출력 유효성을 확인한다.

## 병렬 작업 반환

하위 Agent는 자신이 처리한 관점만 반환한다. 전체 `analyses`를 복사해서 반환하지 않는다. 같은 병렬 라운드에서 같은 관점을 두 번 배정하지 않는다.

```python
return {
    "analyses": {view: result},
    "node_status": {view: result["status"]},
    "references": used_evidence,
    "node_errors": {view: error_message},  # 성공하면 해당 관점만 None으로 갱신
}
```

- `analyses`·`node_status`: 서로 다른 키를 병합하고 같은 키의 최신 결과로 교체한다.
- `references`: source_id별 하나만 유지한다. 같은 출처의 새 발췌는 기존 발췌를 보존하며 합친다.
- `node_errors`: node_status와 동일한 관점 ID로 오류를 조회한다. 병렬 실패를 모두 보존하고, 성공한 작업은 자기 오류만 None으로 갱신한다. 과거 오류 이력은 외부 로그에 남긴다. `last_error`는 제거했다.
- 하위 작업이 모두 합류한 뒤 Supervisor를 실행하는 그래프 연결은 통합 담당자가 구현한다.

## 후보 전환 초기화

`analyses={}`만 반환하면 기존 결과가 지워지지 않는다. 통합 담당자는 `next_candidate`에서 아래처럼 공통 함수를 호출한다.

```python
from ai_investment.state import candidate_reset_update

return {
    "current_idx": next_idx,
    "selected_startup": candidates[next_idx],
    **candidate_reset_update(),
}
```

초기화 함수는 `analyses`, `node_status`, `node_errors`에 LangGraph의 `Overwrite`를 사용한다. 병렬 작업이 끝난 뒤 단일 후보 전환 노드에서만 호출한다.

- 초기화: 관점 결과·상태·오류, 다음 관점, 재작업 횟수·지시, 평가 결과, 초안, 보고서 시도 횟수·경로.
- 유지: 전체 출처·후보 평가 기록, trace_id, 전역 step_count·상한.
- 기존 분석 필드와 profile·점수·투자 판단도 후보 전환 시 초기화한다. 초기화된 decision은 None이다. 새 후보 분석이 완료된 뒤에만 투자 판단을 실행한다. 새 analyses를 기존 투자 판단 입력으로 변환하는 작업은 통합 담당자와 분석 담당자가 연결한다.
- 출처는 실행 전체에서 보존한다. 보고서에는 대상별 결과가 실제 인용한 source_id를 선별한다.
- 기존 `control.select_next_candidate`는 변경하지 않았다. 위 초기화 연결은 통합 담당자의 작업이다.

## 쓰기 권한

`contracts.NODE_WRITES`가 실제 허용 키의 기준이다. `guard_node`를 기존·신규 노드에 적용한다.

- `supervisor`: 라우팅·재작업·단계, 소비한 평가 요청 해제를 위한 eval_result.
- `technical`/`technical_analysis`, `market`/`market_analysis`: 해당 기존 분석 결과와 공통 관점 결과·출처·상태·오류.
- `stakeholder`, `domain`: 공통 관점 결과·출처·상태·오류. 병렬로 같은 business_analysis 필드를 쓰지 않는다.
- `business_analysis`: 기존 business_analysis와 출처. 기존 경로 유지용.
- `next_candidate`: 대상 선택과 공통 초기화 함수가 반환하는 키.
- `report_generation`: report, report_draft.
- `evaluator`: eval_result, report_attempts, resume_after_research.

guard는 최상위 쓰기 키와 하위 Agent의 analyses·node_status·node_errors 관점 키를 검사한다. analyses 키와 결과의 view도 일치해야 한다. 입력 읽기 제한과 State 내부 객체의 직접 변경까지 강제하지는 않는다. 입력 State를 직접 수정하지 않는다.

사업성 기능의 stakeholder·domain 호출과 필수 topic·공통 출력 변환은 [BUSINESS_VIEWS.md](BUSINESS_VIEWS.md)를 따른다. 두 관점은 공통 business_analysis 필드에 동시에 쓰지 않는다.


## 평가 재조사와 횟수 기준

- evaluator가 research를 요청하면 `research_hints[view]`에 구체적인 누락·보완 질문을 넣는다.
- evaluator는 `resume_after_research="report_generation"`을 함께 반환한다. 대상 후보를 유지하고 재조사 후 투자 판단·후보 전환을 반복하지 않는다.
- Supervisor는 target_views와 research_hints를 rework_hints에 반영한 후 소비한 요청을 해제한다. eval_result의 판정·사유는 유지하되 target_views와 research_hints를 비운다.
- Supervisor의 작업 목록이 비면 resume_after_research로 복귀한다. 그래프는 investment_decision과 report_generation 두 경로를 허용해야 한다.
- report_attempts는 평가 미달 횟수다. 초기값 0, 실패마다 +1. 최대 보완 2회라면 1·2번째 미달은 보완하고, 3번째 미달에서 종료한다. 통과하면 횟수와 관계없이 완료한다.
- 재조사도 보완 1회로 계산하고 관점별 재작업 상한을 공유한다. 평가 미달 상한이나 필요한 관점의 재작업 상한이 소진되면 미통과 상태로 종료한다.
- 최종 미통과 사유는 PDF 생성 시점 이후에 생길 수 있다. 종료 전에 최종 eval_result를 반영한 한계 보고서를 다시 출력하고, 품질 통과 보고서와 구분한다. 연결은 보고서·평가·통합 담당자가 구현한다.
- 위 내용은 공통 계약이다. 실제 Supervisor·Evaluator·그래프·최종 PDF 연결은 아직 구현되지 않았다.
