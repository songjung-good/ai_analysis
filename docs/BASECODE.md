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

## 6번 Agent 투자 판단 계약

`agents/decision.py`의 `run(state, model=None)`을 Graph에 연결한다. 테스트에서 모델을 주입할 수 있다. 실제 실행 모델은 프로젝트 `.env`의 `OPENAI_MODEL`이며 `ChatOpenAI`의 구조화 출력을 사용한다. 외부 검색 Tool은 제공하지 않는다.

현재 기업의 기술·사업성·시장 분석을 입력으로 사용한다. 분석이 누락되거나 연결된 출처가 없으면 LLM 호출 없이 보류하고, 출처 ID가 `references`에 없으면 예외를 전파한다. 다른 후보의 누적 평가·점수·출처는 LLM 입력에 넣지 않는다. 출처는 현재 분석의 `source_ids`, `stage_source_ids`로 연결한다.

`decision_models.py`의 `InvestmentAssessment`는 항목 6개의 점수(1~5 또는 `None`), 이유, 출처 ID와 중대한 법률·안전 위험, 핵심 정보 부족을 표현한다. 숫자 점수에는 출처가 필수다. 실제로 출처가 주장을 뒷받침하는지는 모델 및 원문 대조로 별도 확인해야 한다.

`calculate_score`가 가중합을 계산한다. 4.0 이상 `invest`, 3.0 이상 4.0 미만 `conditional`, 나머지는 `hold`다. 점수 누락이면 총점 `None`과 `hold`, 중대한 위험 또는 핵심 정보 부족이면 점수와 무관하게 `hold`다. 판정은 반올림 전 점수로 수행하며 표시할 때만 반올림한다.

반환 키는 `scores`, `investment_score`, `decision`, `decision_reason`, `evaluations`다. `evaluations`에는 현재 기업의 평가 1개만 반환하며 reducer가 누적한다. 기존 필수 이력 4개 키는 유지하고 `score_details`, `blocking_risks`, `missing_information`, `source_ids`를 선택 필드로 추가했다. 새로운 최상위 State 키는 없다.

구현 순서·채점 기준·검증 기록은 [6번 Agent 작업 기록](AGENT6_WORK_PLAN.md)에 있다.

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
