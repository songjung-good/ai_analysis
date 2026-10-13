# Supervisor v2 보고서 - 윤도균 담당

기준: 원격 `AI_Orchestration`의 공통 계약(`72e5aee`), [STATE_CONTRACT.md](STATE_CONTRACT.md),
[업무지침 v2](<261013_agent_업무지침_v2 (1).md>).
공통 `state.py`, `contracts.py`는 가져온 버전을 그대로 사용한다.

## 보고서 노드

```python
from ai_investment.agents.report import run
from ai_investment.contracts import guard_node

builder.add_node("report_generation", guard_node("report_generation", run))
```

기존 `agents.__init__.report_generation`도 같은 `run`을 가리킨다. 보고서 생성은 한 노드에서
PDF와 초안을 함께 반환한다. `report_draft`/`render_pdf`로 노드를 나누지 않는다.
별도 초안 ID, report_feedback, report_evaluation, report_revision 필드를 사용하지 않는다.

반환값은 정확히 다음 두 키다. Counter, 점수, 평가 판정은 수정하지 않는다.

```python
{"report": "PDF의 절대 경로", "report_draft": draft.model_dump(mode="json")}
```

입력:

- `selected_startup`, `domain`, `criteria`, `profile`.
- `analyses[technical|market|stakeholder|domain]`: 공통 `ViewResult`.
- `scores`, `investment_score`, `decision`, `decision_reason`, `evaluations`: 기존 코드 판단.
- `references`: 실행 전체에서 누적되는 Evidence. 현재 관점·profile·대상·현재 평가가 명시적으로
  인용한 ID만 모델에 전달하며 이전 후보의 무관한 출처 인용은 거부한다.
- `report_draft`: 재작성할 이전 초안.
- `eval_result.feedback`, `items`, `research_hints`: 재작성/재조사 피드백.
- `trace_id`: PDF 파일명과 페이지 하단에 기록.

기존 그래프에서 분석의 `source_ids`가 연결되어 있으면 기존 분석 필드를 근거 선별에 사용한다.
공통 analyses가 비어 있다는 사실은 근거 부족 항목에 명시한다. source_id가 없는 과거 기업·
누적 출처로 현재 기업의 사실을 보강하지 않는다.

## ReportDraft 계약

문단은 모두 `{text: str, source_ids: list[str]}`이며 350자/인용 ID 5개 이내다.

| 필드 | 내용 | 상한 |
|---|---|---|
| summary | 핵심 요약 | 2문단 |
| company | 확인된 기업·제품·사업 개요 | 3문단 |
| technical | 핵심 기술·TRL·성능 근거·한계 | 3문단 |
| market | 시장 규모·수요·경쟁 | 3문단 |
| stakeholder | 고객·구매자·공급사·이해 충돌·비용 부담 | 3문단 |
| domain | 적용 현장·도입/PoC·운영 제약 | 3문단 |
| risks | 확인된 위험·불확실성, 장단점 균형 | 3문단 |
| decision_basis | 코드 점수·판정 근거와 한계 | 2문단 |
| missing_information | 코드가 관점 결과에서 추출한 누락·실패·출처 부족 | 필요한 만큼 |
| evaluation_notes | 코드가 붙이는 평가 상태와 최종 미통과 사유 | 필요한 만큼 |

`recommendation`은 `decision_basis`로 교체했다. summary·4관점·risks·decision_basis가
비어 있으면 실패한다. source_ids는 정확히 일치해야 하며 부분 문자열로 복구하지 않는다.
출처가 없어 정보 부족인 문단을 허용하되 평가 통과로 처리하지 않는다. 인용 비율·
출처 다양성·주장 부합성·중립성 판정은 evaluator 소유다.

LLM에 기존 초안과 v2 feedback을 전달한다. temperature=0을 사용하며 코드 점수와 판단을
새로 계산하지 않도록 지시한다. LLM이 인용을 하나도 작성하지 않았는데 현재 근거가
있으면 1회 교정 요청 후 실패한다. 사실성과 중립성은 실제 Judge 평가가 필요하다.

## PDF 및 최종 상태 반영

보통 5쪽 구성이고 내용이 많으면 자동 페이지 분할한다.

1. SUMMARY, 대상, 투자 판단·총점, 분석 상태·보고서 품질, 핵심 요약, 6항목 점수 표, 주요 불확실성 요약.
2. 기업 개요와 기술·시장 관점.
3. 이해관계자·도메인 관점.
4. RISKS 전체 내용, decision_basis와 재검토 조건, 근거 부족 항목 표, 복수 후보의 평가 이력, 품질 평가 상태.
5. REFERENCE: 실제 인용한 출처의 제목·날짜·URL을 표시하고 원문 링크를 제공.

4개 관점은 작성 지시에서 `핵심 판단`, `확인 근거`, `한계` 순서로 구성한다.
기존 문단도 표시할 수 있으며 ReportDraft와 evaluator 계약은 유지한다.
첫 페이지의 위험 요약이 길면 100자 뒤에 말줄임표를 붙이고, 뒤쪽 RISKS에 전체 내용을 표시한다.
긴 원본 decision_reason과 내부 출처 ID는 PDF 본문에 나열하지 않는다.
실행 실패의 상세 오류는 초안과 로그에 유지하고 PDF에는 독자가 이해할 수 있는 안내를 표시한다.

최대 10쪽. 초과하면 오류를 반환하며 문단·이력·참고문헌을 조용히 생략하지 않는다.
임시 PDF를 검증한 뒤 원자적으로 최종 경로에 저장한다. 실패하면 임시 파일을 삭제한다.
파일명은 `investment_report_{trace_id}_{기업명}_{UTC시각}.pdf`이며 반복 생성은 suffix로 구분한다.
표시 날짜는 Asia/Seoul이다. 폰트 자동 탐색 실패 시 `REPORT_FONT_PATH`로 한글 TTF를 지정한다.

후보가 없으면 모델 호출 없이 summary·4관점·risks·decision_basis에 미실시 사유를 작성하고
초안과 PDF를 함께 반환한다. 평가 노드에서 이 경우의 통과/미통과 정책을 결정한다.

`run`은 첫 작성과 보완 작성에서 DRAFT / EVALUATION PENDING으로 표시한다.
이전 초안의 미달을 새 초안의 최종 판정으로 표시하지 않는다.
다만 eval_result.passed=True 또는 passed=False이고 report_attempts>=3이면
내부 finalize로 평가된 초안을 그대로 최종 출력한다. 이 호출에서는 LLM을 실행하지 않는다.

`finalize_report(state)`는 **LLM 없이 평가받은 본문과 인용을 유지**하고, 마지막 eval_result로
평가 상태만 갱신하여 PDF·초안을 함께 반환한다. 통과면 `FINAL / PASS`, 미달이면
`FINAL / NOT PASSED`와 항목별 사유를 표시한다. evaluator verdict가 없으면 호출을 거부한다.
미달 판정을 통과로 바꾸거나 report_attempts를 증가시키지 않는다.

현재 graph.py의 기존 report_generation 노드를 그대로 사용한다. 별도 final_report 노드는 필요 없다.
control.route_after_evaluation은 통과 또는 3번째 미달에서 report_generation을 다시 선택하고,
보고서가 최종 상태를 출력하면 control.route_after_report는 END로 분기한다.
rewrite와 research는 1·2번째 미달에서 기존 보완 경로를 유지한다.
보고서 노드는 eval_result·report_attempts·투자 점수를 수정하지 않는다.
직접 출력 API가 필요한 경우에만 finalize_report를 호출할 수 있다.

최대 보완 2회는 미달 횟수 1·2일 때 재작성/재조사, 3번째 미달에서 최종 한계 보고서라는 뜻이다.
실제 build_graph·Supervisor와 대체 evaluator를 연결해 rewrite 및 research 미달3회 종료,
첫 평가 통과와 보완 후 통과, 두 최종 PDF의 패키징을 검증한다.
테스트는 recursion_limit=100을 명시한다. 운영 CLI의 Supervisor 노드/설정 연결은 팀 통합 범위다.

## 실행·검증

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p '*report*.py' -v
.venv/bin/python examples/run_report_preview.py
```

샘플은 가상 기업·가상 출처를 사용하며 SAMPLE 표시가 들어간다. 실제 Judge나 LangSmith에
연결하지 않으므로 제출용 보고서·트레이스로 사용할 수 없다.
실제 초안 생성에는 OPENAI_MODEL, OPENAI_API_KEY가 필요하다.

## 패키징

실제 최종 PDF·같은 실행의 LangSmith PNG를 받은 뒤 실행한다. 예시 값은 교체해야 한다.

```bash
.venv/bin/python scripts/package_submission.py \
  --repo-url 'https://github.com/songjung-good/ai_analysis/tree/AI_Orchestration' \
  --report output/pdf/FINAL.pdf \
  --traces outputs/tracing-1.png outputs/tracing-2.png \
  --campus CAMPUS --class-name CLASS --names NAME1+NAME2 --run-id ACTUAL_TRACE_ID
```

결과: `output/submission/Agent_{캠퍼스}_{반}_{이름}.zip`.
Git 링크, report.pdf, 순서대로 번호를 붙인 tracing PNG, 실행 ID·SHA-256 manifest만 포함한다.
.env나 저장소 전체를 압축하지 않고 기존 ZIP을 덮어쓰지 않는다.
샘플·초안 PDF·10쪽 초과·필수 섹션 누락·trace_id 불일치·손상된 PNG는 거부한다.
미통과로 종료된 한계 보고서는 최종 상태가 명시되어 있으면 포함할 수 있다.
PNG가 실제 LangSmith 기록인지, 캡처에 재작업/평가 루프가 있는지는 사람이 최종 확인한다.
업로드나 Slack 전송은 실행하지 않는다.
