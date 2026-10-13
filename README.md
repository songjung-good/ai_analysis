# Physical AI·Robotics 투자 평가 — Supervisor 전환

기존 후보 탐색 → 후보별 분석 → 코드 가중합 투자 판단 → PDF 보고서 흐름을 유지하면서
Supervisor 기반 동적 조사와 보고서 품질 평가로 전환하는 프로젝트입니다.

`app.py`는 Supervisor·네 관점·Hybrid evaluator·최종 PDF 출력을 연결한 그래프를 실행합니다.
Mock 검색·Judge와 실제 PDF 렌더러로 진입점을 검증했습니다. 실제 API 실행과 LangSmith
트레이스 캡처는 별도로 확인해야 합니다.

## Overview

- 목표: 후보 기업을 기술·시장·이해관계자·도메인 4관점에서 근거 기반으로 평가합니다.
- 선정 패턴: Supervisor. 부족한 관점만 보완하고 평가 피드백에 따라 재작성·재조사를 선택하는 목적입니다.
- 동적 충분성 기준: status가 done이고 `missing_items`가 비어 있으며 서로 다른 출처 그룹이 2개 이상.
  같은 웹 호스트나 같은 제목의 문서는 하나의 그룹으로 집계합니다. 내용 품질은 Hybrid 평가에서 확인합니다.
- 투자 판단: Python 가중합·Safety Gate를 유지합니다. 보고서에는 점수와 근거·한계를 균형 있게 서술합니다.
- 기준: [업무지침 v2](<docs/261013_agent_업무지침_v2 (1).md>), [공통 State 계약](docs/STATE_CONTRACT.md).

## Features

- 기존 PDF·시장 문헌 RAG와 웹 검색 기반 후보 탐색 및 분석.
- 관점별 최신 결과 병합, 출처 ID별 중복 제거, 노드별 쓰기 허용 필드 검사.
- 보고서 한 노드에서 PDF·구조화 초안 반환, `eval_result.feedback` 반영 재작성.
- 4관점·RISKS·점수 근거·근거 부족 항목, 실제 인용만 REFERENCE로 연결.
- 최종 평가된 본문을 재생성하지 않고 통과/미통과 상태를 PDF에 반영하는 출력 함수.
- 명시된 Git 링크·PDF·트레이스 파일만 ZIP으로 패키징.

확증 편향 통제는 기업 자체 주장과 외부 검증, 장점과 제약을 구분하도록 작성 지시합니다.
Hybrid evaluator가 Groundedness·중립성·편향 통제·관점 커버리지를 판정하고 재작성·재조사를 요청합니다.

## Tech Stack

| 항목      | 구성                                                    |
| --------- | ------------------------------------------------------- |
| Graph     | LangGraph                                               |
| Generator | ChatOpenAI, 모델은 `OPENAI_MODEL`, 보고서 temperature=0 |
| Judge     | Hybrid evaluator, OPENAI_JUDGE_MODEL 또는 OPENAI_MODEL  |
| Retrieval | Chroma, 기술 MMR / 시장 BM25 + Dense similarity         |
| Embedding | `nlpai-lab/KURE-v1`                                     |
| Web       | Tavily                                                  |
| PDF       | ReportLab 생성, PyMuPDF 페이지·본문 검증                |
| Models    | Pydantic + 공통 TypedDict                               |

기존 검색 평가 기록을 재사용합니다. KURE-v1 Hybrid의 **content Recall@5=0.867,
MRR@5=0.671**, 도메인 확인 질문 5개는 모두 Top-3에 정답이 있었습니다.
Hit Rate와 Recall은 평가 정의가 다르므로 해당 값을 별도 Hit Rate 측정값으로 표기하지 않습니다.
이번 보고서 작업에서 검색 평가를 다시 실행하지 않았습니다. [평가 방법·원 결과](docs/MARKET_RETRIEVAL_EVAL.md)

## Agents

| 구성                   | 현재 구현·연결 상태                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| discovery              | 기존 후보 탐색·검증                                                                            |
| profile                | 기존 고객·구매자·도메인 분류                                                                   |
| technical              | 기술 분석·관점 어댑터·재조사 입력 연결                                                         |
| market                 | run_view로 시장 분석·관점 어댑터·재조사 입력 연결                                              |
| stakeholder / domain   | business(view=...)로 관점별 검색·공통 결과·재작업 연결                                         |
| investment_decision    | 사업성 관점을 기존 판단 입력으로 변환. 보고서 재조사 후 동일 후보 점수와 마지막 평가 이력 갱신 |
| report_generation      | v2 `report` + `report_draft` 반환, 4관점·risks·decision_basis, feedback 재작성                 |
| final_report           | 통과·상한 종료 시 finalize_report로 평가된 본문과 최종 상태를 출력                             |
| Supervisor / evaluator | State 기반 동적 관점 배정, Hybrid 평가·재작성·재조사·상한 종료                                 |

보고서 스키마와 evaluator 연동 방법: [SUPERVISOR_REPORT.md](docs/SUPERVISOR_REPORT.md).

## State Schema

공통 파일은 통합 담당자가 관리합니다. 아래는 현재 코드와 제공된 계약의 대응입니다.

| 항목             | 설계·구현                                                                                                                      | 코드/연결 위치                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| 제어 vs 페이로드 | 결과와 제어 메타 블록 분리, `NODE_WRITES`로 쓰기 제한. report는 report·report_draft만 반환                                     | state.py, contracts.py                                  |
| 관측성 위치      | State에 로그 본문을 누적하지 않음. 결정·사유와 검색·평가 로그를 JSONL에 저장; guard는 CLI 진행 출력                            | utils/logger.py, app.py, contracts.py                   |
| 지속성 비용      | analyses는 관점별 최신 결과, references는 source_id 기준 병합. 후보 초기화에 Overwrite 제공                                    | merge_dict, dedupe_by_source_id, candidate_reset_update |
| 상관             | 초기 trace_id를 PDF·JSONL·graph.invoke metadata에 공통 전달                                                                    | state.py, report.py, app.py, utils/logger.py            |
| 재개/복구        | node_status·node_errors로 실패 식별, 성공 시 자기 오류 해제, Supervisor의 제한 재시도. 프로세스 자동 재개는 미지원             | state.py, Supervisor, guard_node                        |
| 동시 처리        | analyses·node_status·node_errors 사전 병합, references dedupe reducer                                                          | state.py                                                |
| 종료 보장        | 관점 재작업 2회·Supervisor 기본20회·보고서 보완2회·후보 max_iterations 상한. 진입점은 이에 맞춘 LangGraph recursion_limit 전달 | state.py, Supervisor/evaluator, control.py, app.py      |

`report_attempts`는 **평가 미달 횟수**입니다. 최초 미달(1), 두 번째 미달(2)은 보완하고,
세 번째 미달(3)에 종료하는 공통 기준입니다. 필요한 관점의 재작업 상한 소진 시 더 일찍 종료할 수 있습니다.
종료 시 통과 여부와 미달 사유를 표시한 최종 PDF를 출력해야 합니다.

## Architecture

진입점은 아래 Supervisor 흐름을 사용합니다. 각 관점의 결과가 합류한 뒤 근거를 판정합니다.

```text
discovery → profile → supervisor → 필요한 관점(병렬) → supervisor
                          └─ 충분/상한 → investment_decision
                                               ├─ 다음 후보 → profile
                                               └─ report_generation → evaluator
                                                    ├─ rewrite → report_generation
                                                    ├─ research → supervisor → 관점 보완
                                                    │                         → 동일 후보 재채점 → report_generation
                                                    └─ 통과/상한 → final_report → END
```

재조사 후 hold/conditional이 되어도 평가 중인 후보를 전환하지 않습니다. 후보 평가 이력은
같은 후보의 최신 점수로 갱신합니다. 최종 출력은 평가한 본문을 재생성하지 않습니다.
후보가 없거나 요청한 관점의 재작업 상한이 소진되면 미통과 한계 보고서로 종료합니다.

이전 RAG 그래프 참고 이미지: [architecture.png](docs/architecture.png).

## Directory Structure

```text
ai_analysis/
├── app.py                       # Supervisor 실행 진입점
├── requirements.txt
├── data/                        # 기술·시장 원문, manifest, 검색 평가 질문
├── docs/                        # 설계·공통 계약·보고서 설명·기존 그래프 이미지
├── examples/
│   ├── run_business_agent.py
│   └── run_report_preview.py    # 외부 API 없는 가상 PDF 미리보기
├── scripts/
│   ├── ingest_tech_docs.py
│   └── package_submission.py   # 명시 파일만 ZIP 생성
├── src/ai_investment/
│   ├── agents/                 # 기존 탐색·profile·분석·decision, v2 report
│   ├── orchestration/          # Supervisor·관점 정의·Hybrid evaluator
│   ├── utils/logger.py         # 실행별 JSONL 로그
│   ├── tools/                  # 웹·RAG 검색
│   ├── state.py                # 공통 State·Reducer·초기화
│   ├── contracts.py            # 공통 NODE_WRITES·guard
│   ├── graph.py                # 동적 관점 배정·평가·최종 출력 그래프
│   ├── control.py
│   ├── scoring.py
│   ├── ingest.py
│   └── retrieval_eval.py
├── tests/                      # 단위 테스트, 대체 Judge를 사용한 연결 검증
├── outputs/                    # 기존 분석 결과; 실제 트레이스 저장 위치
└── output/
    ├── pdf/                    # 생성 보고서 (Git 제외)
    └── submission/             # 제출 ZIP (Git 제외)
```

`orchestration/`과 `utils/logger.py`는 현재 구현·연결되어 있습니다.

## Usage

실행 환경:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env`에 OPENAI_API_KEY, OPENAI_MODEL, TAVILY_API_KEY를 설정합니다.
REPORT_FONT_PATH는 자동 탐색 가능한 한글 폰트가 없을 때 한글 TTF 절대 경로로 지정합니다.
LangSmith 캡처를 사용할 때는 LANGSMITH_TRACING=true, LANGSMITH_PROJECT,
LANGSMITH_API_KEY를 설정합니다. 기존 LANGCHAIN_TRACING_V2·LANGCHAIN_PROJECT 설정도 사용할 수 있습니다.
진입점에서 trace_id metadata와 실행 이름을 전달합니다. `.env`는 app.py와 같은 디렉토리에서 읽습니다.
OPENAI_JUDGE_MODEL은 생략하면 OPENAI_MODEL을 사용합니다.

```bash
# Supervisor 실행: 외부 API 사용, 첫 확인은 후보 1개
python app.py --limit 1 --max-iterations 1

# 기업 직접 지정: 제품·투자 단계는 임의로 채우지 않음
python app.py --company "플라잎" --max-iterations 1

# Supervisor 호출 상한 지정
python app.py --limit 2 --max-iterations 2 --max-steps 20

# 벡터 DB 구축: 임베딩 다운로드 필요
PYTHONPATH=src python -m ai_investment.ingest market
PYTHONPATH=src python scripts/ingest_tech_docs.py

# 외부 API 없는 가상 PDF 미리보기
python examples/run_report_preview.py

# 검증: 실제 API 테스트는 기본 생략
PYTHONPATH=src python -m unittest discover -s tests -v
```

최소 보고서 검증 의존성은 pydantic, reportlab, pymupdf, python-dotenv,
langgraph, langchain-openai이며 전체 테스트에는 langchain-tavily와 langchain-text-splitters도 사용합니다.

최종 PDF와 실제 트레이스를 확보한 뒤 값들을 교체해 실행합니다.

```bash
python scripts/package_submission.py \
  --repo-url 'https://github.com/songjung-good/ai_analysis/tree/AI_Orchestration' \
  --report output/pdf/FINAL.pdf --traces outputs/tracing-1.png outputs/tracing-2.png \
  --campus CAMPUS --class-name CLASS --names NAME1+NAME2 --run-id ACTUAL_TRACE_ID
```

ZIP에는 Git 링크·PDF·번호순 PNG·실행 ID/해시 manifest를 포함합니다. 가상 샘플과 초안 PDF는
제출 도구가 거부합니다. 미통과 종료는 최종 한계 보고서로 구분합니다.
실제 PNG의 재작업·평가 루프, PDF와 캡처가 같은 실행인지 최종 확인해야 합니다.

## Verification

보고서 검증은 v2 입력·정확한 쓰기 계약·feedback 전달·출처 범위·점수 보존·후보 부재·
재작성 연결·최종 통과/미달 상태·10쪽 상한·원자적 출력을 확인합니다.
Mock Judge를 사용한 테스트는 운영 Judge의 품질이나 실제 LangSmith 실행을 증명하지 않습니다.
2026-10-13 진입점 통합 검증: 전체 241개 중 237개 통과, 실제 API 테스트 4개 생략.
진입점 테스트는 실제 그래프·Supervisor·검색 로그·PDF 렌더러를 사용하고 연구·점수·Judge 호출을 Mock으로 대체합니다.
[v2 검증 체크리스트](docs/REPORT_V2_CHECKLIST.md).

## Contributors

기존 README의 개인별 구현 기록을 유지하고 이번 보고서 변경을 반영했습니다.

| 담당   | 기존 구현 및 이번 브랜치에서 확인한 변경                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------------------- |
| 배영환 | 기존 business/decision. 이번 원격 변경의 공통 State·Reducer·NODE_WRITES·후보 초기화 계약                                  |
| 박주윤 | 기존 technical 분석·기술 문서 RAG                                                                                         |
| 이경민 | 기존 market 분석·검색 평가                                                                                                |
| 배은빈 | 기존 profile 분류·고객/구매자 모델                                                                                        |
| 윤도균 | 기존 discovery. 보고서 v2 4관점·decision_basis·초안/PDF 동시 반환·feedback 재작성·최종 평가 상태 출력, README 및 ZIP 도구 |

남은 확인: 실제 API 동적 실행·Judge 내용 판정·LangSmith 캡처·최종 제출 자료 확정.
