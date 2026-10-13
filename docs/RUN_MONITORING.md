# 실행 과정과 시도별 로그 확인

`app.py`는 노드 전체 실행과 내부 작업을 구분해 콘솔과 JSONL에 기록한다.
후보 탐색만 실행하는 `python -m ai_investment.discover`도 로그 파일을 저장한다.

## 기록하는 작업

| 과정 | 내부 작업 |
| --- | --- |
| 후보 선정 | 질의별 후보 검색, Tavily 대기·응답, 기업명 추출, 기업 자격·자료 검증, 채택·제외 사유 |
| 고객 프로필 | 고객·도메인 분류 LLM 호출 |
| 기술 분석 | 검색 질의 생성, 문헌 검색, 관련성 판정, 질의 재작성, 웹 검색, 기술 종합 분석 |
| 시장 분석 | 문헌·웹 검색, 관련성 판정, 질의 재작성, 시장 종합 분석, 경쟁사 근거 정리 |
| 이해관계자·도메인 | 질의별 고객·현장 검색, 사업 관점 구조화 분석 |
| 투자 평가 | 구조화 LLM 응답, 코드 가중 점수·안전 조건 판정 |
| 보고서·품질 평가 | 초안 생성, 본문 LLM 호출, PDF 출력, 품질 규칙 검사, Judge 호출 |

반환 결과가 `failed`이면 노드도 실패로 기록한다. 정상 반환했지만 `missing_items`가
있으면 `insufficient`로 기록하고 콘솔에는 `완료·근거 부족`으로 표시한다.
예외를 던진 호출은 오류를 기록한 뒤 기존 동작대로 예외를 전달한다.
시간 초과·재시도 정책이나 투자 판단 규칙을 변경한 기능은 아니다.

## 로그 파일

```text
outputs/logs/<trace_id>.jsonl
outputs/logs/<trace_id>/attempts/<node>-<attempt_id>.jsonl
```

전체 파일에는 해당 실행의 타임라인을, 시도별 파일에는 한 노드 호출과 그 내부 작업을
저장한다. 같은 기업·노드를 재실행해도 `attempt_id`가 달라 파일을 덮어쓰지 않는다.
각 내부 호출에는 별도 `operation_id`와 상위 `parent_operation_id`가 있다.
병렬 시장 조사에도 실행 문맥을 전달해 기업과 시도별 기록이 섞이지 않게 한다.

주요 필드는 다음과 같다.

- `trace_id`, `node`, `target_name`, `attempt_id`, `node_attempt`: 실행·노드·기업·시도 식별.
- `operation`, `operation_id`, `parent_operation_id`, `event`, `status`: 세부 작업과 시작/종료/실패.
- `timestamp`(UTC), `ts`, `elapsed_ms`: 시각과 소요 시간.
- `query`, `query_index`, `query_count`, `result_count`, `source_ids`, `source_urls`: 검색 내용·진행·출처.
- `model`, `schema`, `input_chars`, `evidence_count`, `timeout_seconds`, `sdk_max_retries`: LLM 호출 설정과 입력 크기.
- `error_type`, `error`, `traceback`: 오류 종류·메시지·파일/행/함수 위치.

프롬프트 본문, 검색 원문 전체, API 키는 별도 저장하지 않는다. `input_chars`는 문자열
또는 직렬화된 메시지의 문자 수이며 토큰 수가 아니다. 모델 설정을 호출 객체에서 얻을
수 없는 경로는 해당 설정을 미확인으로 남긴다.

`node_attempt`는 관점 재작업/보고서 미달 횟수 기반 표시이고, 실제 호출을 구분하는
기준은 `attempt_id`다. `attempt`는 명시적인 호출 재시도 번호다. 기존 business 검색
이벤트는 호환을 위해 전달받은 재작업 횟수(최초 0)를 유지한다.
OpenAI SDK 내부의 자동 재시도는 하나의 LLM 호출 시간에 포함되며, SDK의 각 HTTP
요청을 별도 파일로 분리하지 않는다. Tavily 공통 어댑터의 명시적인 재시도와 discovery의
구조화 출력 교정 시도는 각각 시작/종료 이벤트로 구분한다.

## 확인 명령

```bash
# 실행 중 전체 로그 확인. 시작에 출력되는 실제 실행 ID로 바꾼다.
tail -f outputs/logs/<trace_id>.jsonl

# 완료·실패 작업을 시간과 함께 읽고, 종료 이벤트가 없는 작업은 진행 중으로 표시한다.
python scripts/inspect_run_log.py outputs/logs/<trace_id>.jsonl

# 느린 내부 작업 10개와 아직 종료 기록이 없는 작업 확인.
python scripts/inspect_run_log.py outputs/logs/<trace_id>.jsonl --slowest 10

# 개별 시도의 기록도 같은 도구로 읽을 수 있다.
python scripts/inspect_run_log.py outputs/logs/<trace_id>/attempts/<node>-<attempt_id>.jsonl
```

프로세스를 강제 종료해 종료 이벤트가 기록되지 않으면 진행 중으로 표시될 수 있다.
`Ctrl+C` 등 Python에 전달된 예외는 실패 이벤트에 기록한다. 이전 버전 로그에는 내부
호출별 이벤트가 없으므로 이 도구로 과거 검색/LLM 시간을 복원할 수 없다.

## 후보 목록 작업 이후 1단계가 길어진 이유

기존에는 기업별 제품·투자 검색 2회였다. 현재는 기술 자료와 실제 고객 적용 검색을
더해 기업별 4회다. 질의별 최대 결과 수도 5개로 유지돼 검증 입력량이 늘어날 수 있다.
기준을 통과하지 못한 기업은 제외하고 다음 제안을 확인하므로 실제 검증 기업 수도
최종 채택 기업 수보다 많아질 수 있다.

후보 3개가 모두 바로 통과한다고 가정하면 다음과 같다.

| 경로 | 검색 호출 | 기업명 추출 | 기업별 검증 LLM |
| --- | --- | --- | --- |
| 변경 전 자동 탐색 | 광범위 2회 + 기업별 2회 × 3 = 8회 | 1회 | 3회 |
| 현재 자동 탐색 | 광범위 2회 + 기업별 4회 × 3 = 14회 | 1회 | 3회 |
| 현재 기업 목록 3개 지정 | 기업별 4회 × 3 = 12회 | 생략 | 3회 |

검색과 기업별 검증은 순차 실행한다. 같은 프로세스의 Tavily 요청은 공통 잠금과
호출 간격을 사용한다. 429면 추가 대기·재시도가 발생한다. discovery 모델의
`timeout=60`, `max_retries=2` 설정 때문에 한 논리적 LLM 호출이 60초보다 길 수 있다.
구조화 결과·출처 검증 실패에 대한 교정 요청도 최대 1회 더 실행한다.

`--candidate-list`를 생략하면 목록 파일을 사용하지 않고 자동 탐색한다.
실습용 출발 목록을 사용하려면 다음 명령으로 실행한다.

```bash
python app.py --candidate-list data/discovery_candidates.json --limit 3
```

자료 검증을 유지하면서 소요 시간을 줄일 방법은 새 로그로 가장 오래 걸리는 작업을
확인한 뒤 결정한다. 후보 선정 시간과 후속 분석의 재작업 시간을 함께 비교해야 한다.
