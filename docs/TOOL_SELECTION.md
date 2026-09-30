# AI 스타트업 투자 평가 Agent Tool 선정

## 1. 선정 원칙

- 설계서의 Physical AI·Robotics 투자 평가 흐름을 그대로 지원한다.
- Agent가 직접 선택해야 하는 기능만 Tool로 노출한다.
- 점수 계산과 PDF 생성처럼 실행 순서가 정해진 기능은 일반 Python 함수로 구현한다.
- 모든 검색 결과에 출처 정보를 포함해 최종 보고서의 `REFERENCE`까지 유지한다.
- 제출 일정과 재현성을 고려해 로컬에서 실행 가능한 최소 구성을 사용한다.

## 2. Agent 호출 Tool

외부 Agent Tool은 다음 3개만 사용한다.

| Tool | 구현 기술 | 역할 | 사용 Agent |
|---|---|---|---|
| `web_search` | Tavily | 최신 기업·고객·계약·시장 자료 검색 | 스타트업 탐색, 현장 도입·사업성, CRAG fallback |
| `search_tech_docs` | Chroma + KURE-v1 + MMR | 기술 문서·논문 검색 | 기술·제품 검증 |
| `search_market_docs` | BM25 + Chroma similarity | 시장 보고서·경쟁 자료 하이브리드 검색 | 시장성·경쟁 |

### 2.1 `web_search`

```python
web_search(query: str, domains: list[str] | None = None, max_results: int = 5)
```

- Tavily 검색을 사용한다.
- 기업 홈페이지, 공공기관, 논문, 산업기관 등 신뢰할 수 있는 출처를 우선한다.
- 최대 5개 결과만 반환한다.
- 기술·시장 RAG의 관련성이 낮을 때 CRAG fallback으로 사용한다.

### 2.2 `search_tech_docs`

```python
search_tech_docs(query: str, k: int = 5)
```

- Chroma의 `tech_docs` collection을 검색한다.
- Embedding은 `nlpai-lab/KURE-v1`을 사용한다.
- MMR 설정은 설계서 값을 따른다.

```text
k=5
fetch_k=20
lambda_mult=0.5
```

### 2.3 `search_market_docs`

```python
search_market_docs(query: str, k: int = 5)
```

- Chroma의 `market_docs` collection을 검색한다.
- BM25와 Dense similarity 결과를 결합한다.
- Dense는 MMR 대신 similarity를 쓴다. MMR(`lambda_mult=0.5`)이 정답 청크를 상위 5개 밖으로 밀어내 Recall@5가 0.633에서 0.833으로 오른다(`docs/MARKET_RETRIEVAL_EVAL.md`).

```text
BM25 weight=0.4
Dense similarity weight=0.6
k=5
```

## 3. Tool 공통 반환 형식

모든 검색 Tool은 다음 출처 구조를 반환한다.

```python
class Evidence(BaseModel):
    source_id: str
    title: str
    url: str | None = None
    page: int | None = None
    excerpt: str
    published_at: str | None = None
```

- `source_id`: 중복 제거와 인용 연결에 사용하는 식별자
- `title`: 문서 또는 웹페이지 제목
- `url`: 웹 출처 주소
- `page`: PDF 출처 페이지
- `excerpt`: 판단에 실제 사용한 근거 문장
- `published_at`: 자료 발행일

최종 `references`에는 실제 분석에 사용한 Evidence만 저장한다.

## 4. 내부 함수

다음 기능은 Agent Tool로 노출하지 않고 Graph Node에서 직접 호출한다.

| 함수 | 역할 | 호출 위치 |
|---|---|---|
| `grade_relevance` | 검색 결과와 질문의 관련성 판정 | 기술·제품 검증, 시장성·경쟁 |
| `rewrite_query` | 관련 자료가 부족할 때 검색 질의 재작성 | CRAG 분기 |
| `calculate_score` | 항목별 점수 검증과 가중합 계산 | 투자 판단 |
| `render_report_pdf` | 5페이지 이내 투자 보고서 PDF 생성 | 보고서 생성 |

### 4.1 CRAG 처리

```text
질의 생성
  -> Vector DB 검색
  -> 관련성 평가
  -> 충분: 분석 결과 생성
  -> 부족: 질의 재작성 -> web_search -> 분석 결과 생성
```

### 4.2 점수 계산

LLM은 각 항목의 점수와 근거를 구조화해 반환한다. 최종 가중합은 Python이 계산한다.

```python
WEIGHTS = {
    "team": 0.30,
    "market": 0.25,
    "technology": 0.15,
    "competition": 0.10,
    "traction": 0.10,
    "investment_terms": 0.10,
}
```

- 모든 항목은 1~5 범위인지 검증한다.
- 가중치 합계는 1.0인지 검증한다.
- 정보가 부족한 항목은 임의 점수를 만들지 않고 근거 부족으로 기록한다.

### 4.3 보고서 생성

- LLM은 State에 저장된 분석과 출처만 사용해 보고서 내용을 작성한다.
- ReportLab은 작성된 내용을 PDF로 렌더링한다.
- 첫 장은 `SUMMARY`, 마지막 장은 `REFERENCE`로 구성한다.
- 전체 보고서는 5페이지를 넘지 않는다.

## 5. Agent별 Tool 권한

| Agent | 허용 Tool·함수 |
|---|---|
| 스타트업 탐색 | `web_search` |
| 분야·고객 분류 | Tool 없음, LLM structured output |
| 기술·제품 검증 | `search_tech_docs`, 관련성 부족 시 `web_search` |
| 현장 도입·사업성 | `web_search` |
| 시장성·경쟁 | `search_market_docs`, 관련성 부족 시 `web_search` |
| 투자 판단 | `calculate_score` 직접 호출 |
| 보고서 생성 | `render_report_pdf` 직접 호출 |

각 Agent에는 필요한 Tool만 전달한다. 하나의 Agent에 모든 Tool을 제공하지 않는다.

## 6. RAG 구성

| 구분 | 선정 기술 |
|---|---|
| Document Loader | `PyMuPDFLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Embedding | `SentenceTransformer("nlpai-lab/KURE-v1")` |
| Vector Store | 로컬 영속 Chroma |
| 기술 collection | `tech_docs` |
| 시장 collection | `market_docs` |
| 기술 Retriever | MMR |
| 시장 Retriever | BM25 + Dense similarity |
| Workflow | LangGraph |
| LLM | `ChatOpenAI`, 모델명은 `OPENAI_MODEL` 환경변수로 주입 |
| Web Search | Tavily |
| PDF 생성 | ReportLab |
| 관측성 | LangSmith 선택 적용 |

## 7. 의존성

```toml
[project]
requires-python = ">=3.11,<3.12"
dependencies = [
    "langgraph",
    "langchain",
    "langchain-classic",
    "langchain-openai",
    "langchain-tavily",
    "langchain-community",
    "langchain-chroma",
    "langchain-text-splitters",
    "sentence-transformers",
    "pymupdf",
    "rank-bm25",
    "reportlab",
    "pydantic",
    "python-dotenv",
]
```

## 8. 환경변수

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=
TAVILY_API_KEY=

# 선택 사항
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=ai-startup-investment
```

실제 `.env`는 Git에 추가하지 않는다. 저장소에는 `.env.example`만 포함한다.

## 9. 투자 판단 상태

설계서의 판단 기준과 Graph 상태를 다음과 같이 통일한다.

| 가중 점수 | `decision` | Graph 처리 |
|---:|---|---|
| 4.0 이상 | `invest` | 보고서 생성 |
| 3.0 이상 4.0 미만 | `conditional` | 평가 이력 저장 후 다음 후보 |
| 3.0 미만 | `hold` | 평가 이력 저장 후 다음 후보 |

- 중대한 법률·안전 리스크 또는 핵심 정보 부족은 점수와 관계없이 `hold`로 처리한다.
- 후보가 없거나 `max_iterations`에 도달하면 누적 평가 결과로 보고서를 생성한다.

## 10. 제외 항목

현재 제출 범위에서는 다음 기술을 사용하지 않는다.

- MCP
- Supervisor 전용 라이브러리
- OCR·VLM Parser
- Cross-Encoder reranker
- 별도 Chroma 서버
- PostgreSQL·Redis checkpoint
- Streamlit UI
- 별도 계산 Agent

문서 추출 실패, 검색 성능 부족, 장기 실행 상태 보존 같은 문제가 실제로 확인될 때만 추가한다.
