# 보고서 생성 Agent

구현: `src/ai_investment/agents/report.py`

## 입력과 결과

`run(state)`는 후보·분석·점수·판단·평가 이력·출처를 읽고 `{"report": "PDF의 절대 경로"}`만 반환한다.
외부 검색 Tool은 받지 않는다. `OPENAI_MODEL`에 지정한 ChatOpenAI 모델이 State와 Evidence로 짧은 보고서 문단을 작성한다.
점수, 판단, 기업 기본 정보는 LLM 문장에서 추출하지 않고 State 값을 직접 PDF에 표시한다.

보고서는 `output/pdf/`에 생성한다. 같은 기업을 반복 평가해도 파일을 구분하도록 UTC 시각을 파일명에 넣는다.
최종 PDF는 임시 파일에 작성한 뒤 페이지 수와 첫·마지막 페이지 제목을 검증하고 원자적으로 이동한다.
문제가 발생하면 임시 파일을 삭제하고 오류를 전달한다.

## 페이지 구성

평가할 후보가 있으면 다음 5페이지를 생성한다.

1. `SUMMARY`: 대상 기업, 최종 판단·점수, 핵심 이유와 요약. 내용은 반 페이지 이내.
2. 기업 및 사업 개요: 제품, 투자 단계, 팀, 사업 내용.
3. 기술 및 시장 분석: State에 저장된 기술·시장 분석 요약.
4. 투자 평가 및 최종 의견: 점수표, 최대 10개 평가 이력, 후속 확인 사항.
5. `REFERENCE`: 보고서 문단이 실제로 인용한 출처만 기재.

후보가 없으면 LLM을 호출하지 않고 `SUMMARY`와 `REFERENCE`로 된 2페이지 보고서를 생성한다.
요약이나 최종 의견이 비어 있거나, 출처 ID가 State에 없거나, 한 페이지에 내용이 다 들어가지 않으면 오류를 반환한다.
State에 출처가 있지만 초안에 인용이 하나도 없으면 모델에 한 번 교정을 요청하고, 계속 인용이 없으면 오류를 반환한다.
내용이나 참고문헌을 조용히 잘라내지 않는다.

## 설정

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=...
# 선택: 자동 검색되는 한글 폰트가 없는 운영체제에서 지정
REPORT_FONT_PATH=/absolute/path/to/KoreanFont.ttf
```

기본 탐색 폰트는 macOS의 AppleGothic/Arial Unicode와 Linux의 Nanum/Noto CJK다.
PDF 생성에는 기존 `requirements.txt`의 `reportlab`, 렌더링 검증에는 `pymupdf`가 필요하다.

## 단독 호출

```python
from ai_investment.agents.report import run

update = run(state)  # state에는 selected_startup, 분석, 판단, references 등이 들어간다.
print(update["report"])
```

이전 후보의 상세 분석은 현재 GraphState에 보존되지 않는다. 따라서 보고서는 현재 선택된 기업을 자세히 다루고,
이전 후보는 `evaluations`에 남은 이름·점수·판단만 요약한다. 모든 후보를 상세 비교하려면
`Evaluation`에 후보별 분석과 출처를 저장하도록 State 계약을 확장해야 한다.

## 검증

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
