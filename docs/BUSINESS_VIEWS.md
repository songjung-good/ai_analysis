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

Step 3의 관점별 검색·프롬프트 분리, Step 4의 공통 출력 변환, Step 5의 재작업 반영과 검색 로그를 구현했다. 실제 API·Supervisor 통합 실행을 검증했다는 의미는 아니다.

## 재작업 입력과 결과

- Supervisor는 `rework_hints[view]`에 누락 항목·보완 질문을 문자열 목록으로 전달한다.
- 공백·중복·빈 요청을 정리하고 해당 관점의 요청만 읽는다. 다른 관점의 요청은 검색과 분석에 섞지 않는다.
- 요청이 있으면 기업·관점·요청 내용 중심으로 질의를 바꾼다. 고객 사례, 독립 검증·반대 근거, 공식 조건·위험의 세 방향으로 검색한다. 호출 수는 3회, 호출당 결과 상한은 5개다.
- 요청은 분석 payload의 rework_hints에도 전달한다. 요청이 해결됐다는 가정으로 missing_items를 자동 삭제하지 않는다.
- 재작업 때 자기 관점의 이전 done 결과와 그 결과가 인용한 Evidence만 재사용한다. 과거의 다른 관점·전체 출처를 사실 근거로 넘기지 않는다.
- 모델은 기존 근거와 새 근거를 검토해 최신 전체 관점 결과를 반환한다. 기존 결과는 독립 근거로 취급하지 않는다. 반대 근거는 반영하고 미해결 요청·핵심 상충은 view_missing_items에 남긴다.
- 이전 결과의 대상 일치는 후보 전환 시 candidate_reset_update()를 적용하는 그래프 계약으로 보장해야 한다. 이 연결은 통합 담당자 작업이다.
- 검색이 비어 있고 재사용할 근거도 없으면 모델을 호출하지 않고 요청과 필수 누락을 반환한다.
- Agent는 rework_hints·횟수를 직접 수정하지 않는다. 요청 소비와 상한 관리는 Supervisor가 담당한다.

## 검색 로그 연결

표준 logging의 `ai_investment.agents.business_search` 로거가 INFO 레코드를 생성한다. 자체 파일 Handler나 별도 logger 모듈은 추가하지 않았다.

- event: search_started, search_finished, search_failed.
- 연결 필드: trace_id, target_name, node, view, attempt(관점별 재작업 횟수).
- 작업 필드: query, rework_hints, result_count(완료), error(실패).
- query는 실행 직전에 기록한다. 첫 호출 실패 시 실행하지 않은 나머지 질의를 실행 기록으로 남기지 않는다.
- 통합 담당자는 INFO 레벨을 활성화하고 JSONL Handler/Formatter에서 위 extra 필드를 저장한다. 표준 logging 레코드는 설정 없이 JSONL·LangSmith에 자동 저장되지 않는다.
- logger 연결은 [STATE_CONTRACT.md](STATE_CONTRACT.md)의 trace_id·외부 관측성 계약을 따른다. Mock 검증 로그는 실제 실행 Trace로 제출하지 않는다.
