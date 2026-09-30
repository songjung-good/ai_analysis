# 스타트업 탐색 Agent

구현: `src/ai_investment/agents/discovery.py`

## 역할과 처리 순서

1. `domain`과 `criteria`를 검증한다.
2. 영문·국문 검색 질의로 후보 자료를 수집한다.
3. LLM structured output으로 출처가 있는 기업명을 추출한다.
   기업명 뒤에 설명이 붙으면 이름만 분리한다.
4. 기업별로 제품·창업자·최근 투자 라운드, 상장·인수·합병 여부를 추가 검색한다.
5. 출처가 연결된 정보로 선정 조건을 확인한다. 조건 불일치와 핵심 정보 부족은 제외한다.
6. 중복 기업을 제거하고 후보 목록, 첫 후보, 실제 사용한 출처를 반환한다.

외부 Tool은 기존 `TOOLS`에 등록된 `web_search`만 사용한다. 점수 계산이나 투자 판단은 수행하지 않는다.
LLM은 `OPENAI_MODEL`로 지정하고, [LangChain structured output](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output)을 사용한다.

## 입력 계약

`domain`은 비어 있지 않은 문자열이다. `criteria`는 아래 키만 지원한다. 오타나 미지원 키는 오류로 처리한다.

| 키 | 기본값 | 의미 |
|---|---|---|
| `region` | `None` | 기업 소재 지역. `None`이면 전 세계 |
| `funding_stages` | `["Seed", "Series A", "Series B", "Series C"]` | 허용하는 최근 투자 단계. 이 네 값 중 하나 이상 |
| `candidate_limit` | `5` | 반환 후보 상한, 1~10 |
| `disclosure_requirement` | `None` | 추가 자료 공개 조건. 예: `"공개된 고객 도입 사례"` |

도메인 적합성, 비상장 상태, Exit 미완료는 필수 조건이다. 제품과 허용 투자 단계도 근거가 있어야 한다.
기업별 선정 조건 중 하나라도 불명확하면 후보에서 제외한다. 팀 정보만 없으면 `team=[]`로 전달한다.
회사 홈페이지에 인수 소식이 없다는 이유만으로 Exit 미완료를 확정하지 않도록 프롬프트에 명시했다.
최근 독립 스타트업으로서의 투자 유치와 독립 운영 근거가 함께 있고 반대 자료가 없으면 두 출처를 인용해 판단한다.

`max_iterations`는 공통 Graph의 평가 반복 상한이다. 탐색 후보 수는 `candidate_limit`로 따로 설정한다.

## 출력 계약

`run(state)`는 입력 State를 변경하지 않고 다음 키만 반환한다.

```python
{
    "candidates": [
        {"name": "기업명", "product": "제품 설명", "funding_stage": "Series A", "team": []}
    ],
    "current_idx": 0,
    "selected_startup": candidates[0],
    "references": [Evidence(...)],
}
```

위 예시는 반환 구조 설명용이며 실제 기업 정보가 아니다.

`references`에는 채택한 기업의 정보와 선정 조건을 뒷받침하는 자료만 추가한다.
LLM이 반환한 모든 출처 ID가 실제 검색 결과에 있는지 검사한다. 출처 없는 사실과 검색에 없는 ID는 오류로 처리한다.
같은 자료가 여러 검색에 나타나면 출처 ID를 기준으로 합치되, 서로 다른 발췌문은 보존한다.
출처 ID 검증은 인용 대상의 존재를 확인한다. 문장과 판단의 의미적 일치, 웹 자료의 최신성과 사실성은 LLM 검증과 원문 검토에 의존한다.

후보가 0개이면 `candidates=[]`, `current_idx=0`, `selected_startup={}`, `references=[]`를 반환한다.
공통 Graph는 이 경우 분석·투자 판단을 건너뛰고 보고서 Node로 이동한다.
보고서 Agent는 빈 후보·빈 평가 이력을 받아 2페이지 PDF를 생성한다. 전체 흐름의 실제 API 기반 실행은 별도로 검증해야 한다.

## 실패 처리와 호출 상한

- 검색은 질의당 최대 2회 시도한다. 잘못된 입력과 비일시적 HTTP 오류는 즉시 전달한다.
- 검색 장애와 응답 형식 오류를 후보 없음으로 바꾸지 않는다.
- 구조화 응답이나 출처 검증 실패는 한 번 교정 요청한 뒤 계속 실패하면 오류를 전달한다.
- LLM 클라이언트의 요청 제한 시간은 60초이고, 클라이언트 재시도는 최대 2회이다.
- 후보 제안 검증은 최대 `min(candidate_limit * 2, 20)`개이다.
- 검색 질의 수는 최대 `2 + 2 * min(candidate_limit * 2, 20)`개이며, 각 질의는 최대 5개 결과를 받는다. 재시도는 별도이다.
- 이 상한 안에서 충분한 근거를 찾지 못하면 요청한 수보다 적은 후보 또는 빈 목록을 반환할 수 있다.

## 단독 실행

Python 3.11 환경에서 실행한다. 전체 프로젝트 의존성은 `requirements.txt`에 있고, 탐색 단독 실행에 필요한 패키지는 다음과 같다.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install pydantic langchain-openai langchain-tavily python-dotenv
```

로컬 `.env`에 다음 값을 설정한다. API 키가 들어간 `.env`는 Git에 추가하지 않는다.

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=your-structured-output-capable-model
TAVILY_API_KEY=your-tavily-api-key
```

저장소 루트에서 실행하면 후보와 출처가 JSON으로 출력된다.

```bash
PYTHONPATH=src .venv/bin/python -m ai_investment.discover --region 대한민국 --limit 3
```

IDE의 파일 실행 기능이나 파일 경로로도 같은 CLI를 실행할 수 있다. 이 방식은 `PYTHONPATH` 설정이 필요 없다.

```bash
.venv/bin/python src/ai_investment/agents/discovery.py --region 대한민국 --limit 3
```

코드에서 호출할 때는 다음과 같이 설정한다.

```python
from ai_investment.agents.discovery import run
from ai_investment.state import create_initial_state

state = create_initial_state(
    domain="Physical AI/Robotics",
    criteria={"region": "대한민국", "candidate_limit": 3},
    max_iterations=3,
)
update = run(state)
```

## 검증

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Graph 통합 테스트에는 `langgraph`도 필요하다. 테스트는 검색·LLM을 모의 객체로 바꾸므로 API 키와 네트워크가 필요 없다.
정상 반환, 선정 조건, 근거 부족, 중복 제거, 호출 상한, 출처 검증, 오류 전달·재시도, 빈 후보의 Graph 분기를 검증한다.
실제 API 검색 품질은 환경변수를 설정한 뒤 단독 실행 결과의 출처를 검토해 확인한다.
