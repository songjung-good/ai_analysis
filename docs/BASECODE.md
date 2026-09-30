# Basecode 개발 계약

## 목적

각 Agent를 별도 파일에서 개발하면서 병렬 분석 과정의 State 충돌을 막습니다.

## 파일 소유권

| Agent | 구현 파일 | 쓸 수 있는 State 키 |
|---|---|---|
| 스타트업 탐색 | `agents/discovery.py` | `candidates`, `current_idx`, `selected_startup`, `references` |
| 분야·고객 분류 | `agents/profile.py` | `profile`, `references` |
| 기술·제품 검증 | `agents/technical.py` | `technical_analysis`, `references` |
| 현장 도입·사업성 | `agents/business.py` | `business_analysis`, `references` |
| 시장성·경쟁 | `agents/market.py` | `market_analysis`, `references` |
| 투자 판단 | `agents/decision.py` | `scores`, `investment_score`, `decision`, `decision_reason`, `evaluations` |
| 보고서 생성 | `agents/report.py` | `report` |

`guard_node`가 계약 밖 State 쓰기를 실행 중 차단합니다. 병렬 Agent가 함께 쓰는 `references`와 반복 평가에서 누적하는 `evaluations`만 reducer로 병합합니다. `references`의 각 항목은 `Evidence` 모델을 사용합니다.

## 구현 규칙

1. `run(state)`는 State 전체가 아닌 변경한 키만 `dict`로 반환합니다.
2. 다른 Agent 소유 키를 수정하지 않습니다.
3. 분석 근거로 실제 사용한 자료만 `references`에 추가합니다.
4. 예외를 빈 결과로 숨기지 않습니다. 재시도와 fallback은 해당 Agent 내부에서 끝냅니다.
5. 공용 State 변경이 필요하면 `state.py`, `contracts.py`, 설계서를 함께 수정합니다.

## 4번 Agent 입력 계약

`selected_startup`은 기존 `Startup` 구조를 사용한다. `profile`은 분야·고객 분류 Agent가 다음 구조로 전달한다. 최상위 State 키는 변경하지 않는다.

| 입력 필드 | 타입 | 4번 Agent 처리 |
|---|---|---|
| `selected_startup.name` | `str` | 필수. 누락·빈 문자열·공백만 있는 값은 `ValueError`. 문자열이 아니면 `TypeError` |
| `selected_startup.product` | `str` | 누락·빈 값이면 정보 부족 기록, 기업명으로 검색 |
| `selected_startup.funding_stage` | `str` | 누락·빈 값이면 정보 부족 기록, 분석 계속 |
| `selected_startup.team` | `list[str]` | 누락·빈 목록이면 정보 부족 기록, 분석 계속 |
| `profile.subdomain` | `str` | 세부 분야. 누락·빈 값이면 정보 부족 기록 |
| `profile.paying_customer` | `str` | 비용을 지불하는 고객 유형. 누락·빈 값이면 정보 부족 기록 |
| `profile.customer_problem` | `str` | 고객의 현장 문제. 누락·빈 값이면 정보 부족 기록 |
| `profile.info_sufficiency` | `str` | 상위 Agent의 정보 충분성 설명. 전달된 값을 보존하며 이 값만으로 분석 중단 또는 투자 판단하지 않음 |
| `profile.missing_info` | `list[str]` | 상위 Agent가 기록한 부족 정보. 4번 Agent가 발견한 누락과 함께 중복 제거해 기록 |

`selected_startup`이 없으면 기업명 누락으로 처리한다. `profile`이 없으면 분야·고객 정보 부족으로 기록하고 기업명으로 검색한다. 선택 필드의 `null`은 누락으로 처리하며 값이 제공되었는데 타입이 다르면 `TypeError`로 처리한다. 문자열은 앞뒤 공백을 제거하고, 목록은 각 문자열 항목의 타입을 확인한다. 원본 State는 변경하지 않는다.

`info_sufficiency`의 가능한 값은 아직 enum으로 제한하지 않는다. 상위 Agent의 `충분` 선언과 무관하게 실제 필드 누락을 검사한다. 입력 부족은 향후 `business_analysis.information_gaps`에 기록하며 분석 결과의 정보 부족과 구분해 `입력 누락: 필드명` 형태로 표시한다.

제품명 또는 고객 정보가 부족하면 첫 검색은 기업명 중심으로 구성하고, 확보한 근거에 따라 추가 질의를 구체화한다. 병렬 기술·시장 Agent의 결과를 기다리지 않는다.

입력 예시는 `tests/fixtures/business_input_complete.json`, `tests/fixtures/business_input_name_only.json`에 있다. 가상 기업 데이터이며 검색 근거나 분석 완료 결과를 의미하지 않는다. 입력 검증 코드는 4번 Agent 분석 로직 구현 단계에서 이 계약을 적용한다.

## Tool 권한

각 Agent 파일의 `TOOLS`에는 허용된 외부 Tool만 들어갑니다.

| Agent | 외부 Tool |
|---|---|
| 스타트업 탐색 | `web_search` |
| 분야·고객 분류 | 없음 |
| 기술·제품 검증 | `search_tech_docs`, `web_search` |
| 현장 도입·사업성 | `web_search` |
| 시장성·경쟁 | `search_market_docs`, `web_search` |
| 투자 판단 | 외부 Tool 없음, `calculate_score` 직접 호출 |
| 보고서 생성 | 외부 Tool 없음 |

## Graph 제어

- 분야·고객 분류 후 기술, 사업성, 시장 Agent가 병렬 실행됩니다.
- 세 Agent가 모두 끝난 뒤 투자 판단을 실행합니다.
- `invest`면 보고서를 생성합니다.
- `conditional` 또는 `hold`이며 후보와 반복 횟수가 남으면 다음 후보를 선택합니다.
- 후보 소진 또는 `max_iterations` 도달 시 보고서를 생성합니다.

## 확인

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
