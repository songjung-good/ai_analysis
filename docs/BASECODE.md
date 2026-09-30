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

- 탐색 결과 후보가 없으면 분석을 건너뛰고 보고서를 생성합니다.
- 분야·고객 분류 후 기술, 사업성, 시장 Agent가 병렬 실행됩니다.
- 세 Agent가 모두 끝난 뒤 투자 판단을 실행합니다.
- `invest`면 보고서를 생성합니다.
- `conditional` 또는 `hold`이며 후보와 반복 횟수가 남으면 다음 후보를 선택합니다.
- 후보 소진 또는 `max_iterations` 도달 시 보고서를 생성합니다.

## 확인

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
