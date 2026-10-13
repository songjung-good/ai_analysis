# 사업성 관점별 실행·출력 계약

## 호출

기존 `business_analysis(state)` 호출은 기존 BusinessAnalysis 결과와 references를 반환한다. 새 관점은 같은 함수에 view를 지정한다.

```python
from functools import partial
from ai_investment.agents import business_analysis
from ai_investment.contracts import guard_node

stakeholder_node = guard_node("stakeholder", partial(business_analysis, view="stakeholder"))
domain_node = guard_node("domain", partial(business_analysis, view="domain"))
```

두 노드는 analyses의 자기 관점, 해당 관점이 인용한 references, node_status, node_errors만 반환한다. 공통 business_analysis 필드에는 쓰지 않는다. 실제 Supervisor·그래프 연결은 통합 담당자가 수행한다.

## 필수 확인 항목

BusinessAnalysis의 `view_findings`는 topic, text, source_ids를 가진 목록이다. 각 항목은 제공된 근거를 인용해야 하며, 없는 주제를 추정으로 채우지 않는다.

| 관점 | 허용·필수 topic |
| --- | --- |
| stakeholder | customer_buyer, user, supplier, conflicts, cost_burden |
| domain | application, deployment, constraints |

- 이해관계자: text에 주체의 이름·역할·부담·조건을 확인한 범위에서 기록한다.
- 도메인 적용: text에 제품·현장·시점·운영 단계·유료 여부·제약 조건을 확인한 범위에서 기록한다.
- conflicts에는 확인된 충돌이나 근거가 있는 검토 결과를 기록한다. 자료를 찾지 못했다고 ‘충돌 없음’으로 작성하지 않는다.
- `view_missing_items`에는 해당 관점의 미확인 항목과 핵심 상충을 기록한다.
- 기존 `information_gaps`는 입력·조사 이력이다. 이를 그대로 Supervisor의 재작업 조건으로 사용하지 않는다.

## 공통 결과 변환

`business.to_view_result()`가 다음과 같이 변환한다.

- view_findings의 text·source_ids를 ViewResult.claims로 변환한다.
- 인용한 source_id의 합집합을 ViewResult.source_ids로 반환한다.
- view_missing_items와 빠진 필수 topic을 missing_items로 반환한다.
- 다른 관점의 topic은 오류로 처리한다.
- 최소 항목이 존재해도 내용의 충실성·중립성·출처 독립성이 자동 보장되지는 않는다. Supervisor와 Hybrid 평가에서 확인한다.

source_ids는 제공된 Evidence와 대조한다. 공백과 8자 이상의 유일한 ID 접두어만 보정하며, 모호한 접두어나 알 수 없는 ID는 오류로 처리한다. 인용 필드 외의 설명·정보 부족 목록은 보정하지 않는다.

## 부족과 실패

- 검색 결과 없음: 작업 status는 done, claims·source_ids는 비어 있고 필수 누락을 반환한다. node_errors는 None이다.
- 검색·모델·출력 검증 실패: 해당 관점 status는 failed, node_errors에 오류를 반환한다. 성공 결과처럼 사용하지 않는다.
- 실행 성공: 해당 관점 node_errors를 None으로 갱신한다. 다른 관점의 오류는 지우지 않는다.
- view·대상 입력 오류는 외부 호출 전에 예외로 반환한다. 기존 view 없는 호출의 예외 동작은 유지한다.
- 입력 State를 직접 수정하지 않는다. 이전 분석과 출처는 호출의 사실 근거로 임의 사용하지 않는다.

## 구현 단계

Step 3의 관점별 검색·프롬프트 분리와 Step 4의 공통 출력 변환을 구현했다. `rework_hints`에 따른 질의 변경과 실제 질의 로그는 Step 5에서 구현한다. 현재 실제 API·Supervisor 통합 실행을 검증했다는 의미는 아니다.
