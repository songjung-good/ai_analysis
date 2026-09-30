# AI 스타트업 투자 평가 시스템

LangGraph 기반 Multi-Agent와 Agentic RAG를 활용해 AI 스타트업의 기술력, 시장성, 경쟁력을 분석하고 투자 가능성을 평가하는 프로젝트입니다.

# AI Startup Investment Evaluation Agent

본 프로젝트는 Physical AI·Robotics 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview

- Objective : Physical AI·Robotics 스타트업의 창업자·팀(30%), 시장성(25%), 제품·기술력(15%), 경쟁 우위(10%), 실적(10%), 투자 조건(10%)을 기준으로 투자 적합성 분석
- Method : LangGraph Multi-Agent, Agentic RAG(CRAG), LLM 구조화 출력 + Python 점수 계산

## Features

- 웹 검색 기반 후보 스타트업 탐색 및 선정 조건(비상장, Seed~Series C, Exit 미완료) 검증
- 세부 분야·지불 고객·고객 문제 분류 (정보가 없으면 지어내지 않고 "부족"으로 표시)
- PDF 자료 기반 기술·시장 정보 추출 (CRAG: 벡터 검색 → 관련성 평가 → 질의 재작성 → 웹 검색 폴백)
- 기술·제품 검증, 현장 도입·사업성, 시장성·경쟁 분석 3개 에이전트 병렬 실행
- 출처 ID 검증: LLM이 인용한 출처가 실제 검색 결과에 있는지 확인하고, 실제 인용한 자료만 REFERENCE에 기재
- 가중치 기반 투자 점수 산출 및 판단(추천 / 조건부 검토 / 보류), 필수 보류 조건 적용
- 보류 시 다음 후보로 넘어가는 반복 평가
- 5페이지 이내 투자 보고서 PDF 생성 (SUMMARY로 시작, REFERENCE로 끝남)

## Tech Stack

- Framework : LangGraph
- LLM/Generator : OpenAI ChatOpenAI ({GPT version}, `OPENAI_MODEL` 환경변수로 지정)
- LLM/Judge : {GPT version} (관련성 평가·투자 판단에 동일 모델 사용)
- Retrieval : Chroma - 기술 문서(MMR, top_k=5) / 시장 문서(BM25+Dense 하이브리드), 시장 문서 Recall@5 0.867
- Embedding : nlpai-lab/KURE-v1
- Web Search : Tavily
- Report : ReportLab(PDF 생성), PyMuPDF(페이지 수·제목 검증)

## Agents

- 1. 스타트업 탐색 : 웹 검색으로 후보를 추출하고, 기업별 추가 검색으로 투자 단계·비상장·Exit 여부를 근거와 함께 검증. 중복 제거, 출처 ID 검증, 검색 호출 상한 적용
- 2. 분야·고객 분류 : LLM 구조화 출력으로 세부 분야(8개 목록), 지불 고객, 고객 문제를 정의. 제품 설명이 없으면 LLM 호출 없이 "정보 부족" 처리
- 3. 기술·제품 검증 : 논문·기술 보고서 RAG(CRAG)로 핵심 기술, 성능 근거, 한계를 검증. 기업 성능과 업계 벤치마크를 구분하고, 문헌만 인용하면 근거 수준을 `low`로 제한
- 4. 현장 도입·사업성 : 웹 검색으로 시연/시험 운영/유료 운영을 구분하고 고객 사례, 도입 비용·효과, 규제 위험, 근거 수준을 분석
- 5. 시장성·경쟁 : 시장 문서 RAG(CRAG)로 시장 규모·수요·경쟁을 병렬 검색. 경쟁사는 코드가 선정해 같은 입력에 같은 결과를 보장하고, 근거에 없는 수치·기업명은 제거
- 6. 투자 판단 : LLM이 항목별 점수와 근거를 만들고 Python이 가중합과 최종 판단을 계산. 자료가 없는 항목은 1점이 아니라 `None`으로 처리하고 보류
- 7. 보고서 생성 : State의 분석·점수·출처로 5페이지 PDF 작성. 인용하지 않은 자료는 REFERENCE에서 제외

## Architecture

![Architecture](docs/architecture.png)

### RAG 서브그래프 (CRAG)

![CRAG](docs/crag.png)

## Directory Structure

```
├── data/                  # 문서 풀 (기술 문서, 시장 문서)
├── scripts/               # 문서 적재 스크립트 (Chroma)
├── src/ai_investment/
│   ├── agents/            # 평가 기준별 Agent 모듈
│   ├── tools/             # RAG, 웹 검색 등 Agent별 도구
│   ├── graph.py           # LangGraph 흐름 정의
│   ├── state.py           # 공유 State 정의
│   └── scoring.py         # 투자 점수 계산
├── docs/                  # Agent별 설계·작업 문서
├── tests/                 # 단위 테스트
├── output/pdf/            # 생성된 보고서
└── README.md
```

## Usage

### 1. 실행 환경 준비

저장소 루트에서 Python 3.11과 uv로 설치합니다. 기존 `.venv`가 있으면 생성은 생략합니다.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

`.env`가 없으면 `.env.example`을 복사하여 `OPENAI_API_KEY`, `OPENAI_MODEL`, `TAVILY_API_KEY`를 설정합니다. 이미 구성한 `.env`는 덮어쓰지 않습니다. 기본 DB 설정은 `CHROMA_PERSIST_DIRECTORY=data/chroma`입니다. 상대 경로는 저장소 루트를 기준으로 해석합니다.

### 2. 원본 PDF 준비

기술 자료는 `data/tech_docs/manifest.json`의 `file` 이름과 일치하도록 `data/tech_docs/`에 준비합니다. 사용할 페이지 범위도 이 manifest에서 지정합니다.

시장 자료는 다음 디렉터리를 만든 뒤 `data/market/sources.json`과 동일한 파일명으로 원본 PDF 3개를 넣습니다.

```bash
mkdir -p data/market/raw
```

```text
data/market/raw/
├── startupalliance_2026_physical_ai_startup_map.pdf
├── spri_is202_physical_ai_status.pdf
└── ifr_world_robotics_2025_press.pdf
```

원본은 각각 스타트업얼라이언스의 「2026 피지컬 AI 스타트업맵」, 소프트웨어정책연구소의 「피지컬 AI의 현황과 시사점(IS-202)」, IFR의 「World Robotics 2025 Press Conference」입니다. `data/market/transcripts/`는 텍스트 추출을 보완하는 자료이며 원본 PDF를 대체하지 않습니다. PDF가 누락되면 필요한 파일명과 준비 경로를 오류로 안내합니다.

### 3. Chroma 구축 — 최초 1회 또는 문서 변경 시

먼저 입력 자료를 확인합니다. `--dry-run`은 임베딩·DB 적재를 수행하지 않습니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_tech_docs.py --dry-run
PYTHONPATH=src .venv/bin/python -m ai_investment.ingest market --dry-run
```

확인이 끝나면 기술·시장 컬렉션을 각각 구축합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_tech_docs.py
PYTHONPATH=src .venv/bin/python -m ai_investment.ingest market
```

Chroma는 로컬 영속 DB이므로 별도 서버를 실행할 필요가 없습니다. KURE-v1 모델 최초 다운로드에는 네트워크와 시간이 필요합니다.

주의: 적재 명령을 다시 실행하면 해당 컬렉션의 기존 데이터가 초기화됩니다. 검색 실행 시 자동 구축·재구축하지 않습니다. DB 또는 필요한 컬렉션이 없거나 비어 있으면 구축 명령을 안내하고 중단합니다. 원본 자료를 준비하지 않은 상태에서 빈 DB를 정상적인 분석 결과로 처리하지 않습니다.

### 4. 실행 및 테스트

현재 전체 분석용 `app.py`는 없습니다. 전체 실행은 `AgentNodes`로 각 에이전트를 연결하여 `build_graph`를 호출해야 합니다. 단독 실행 예제는 다음과 같습니다. 실제 실행은 검색·LLM API 비용이 발생합니다.

```bash
# 1번: 스타트업 탐색
PYTHONPATH=src .venv/bin/python -m ai_investment.discover --region 대한민국 --limit 3

# 4번: 현장 도입·사업성 예제
.venv/bin/python examples/run_business_agent.py

# 기본 테스트: 실제 API 테스트는 생략
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Results

- 임베딩 비교(KURE-v1, bge-m3, multilingual-e5-large) 결과 차이가 1문항뿐이어서 설계대로 KURE-v1 유지
- 시장 문서 검색 개선(Dense를 MMR에서 similarity로 변경, 소제목 경계 청킹): Recall@5 0.633 → 0.867, 도메인 질문 5개 모두 상위 5개 안에서 검색
- 경쟁사 선정을 코드로 고정: 같은 입력 3회 실행 시 3회 동일 결과 (이전 6회 6가지)
- 시장성·경쟁 실행 시간 168초 → 30~50초
- 테스트: 전체 150개 중 146개 통과, 실제 API를 쓰는 기술 Agent 스모크 테스트 4개 생략 (데이터 적재·DB 사전 검사 변경 검증 기준)

## Limitations

- 근거 부족 시 해당 에이전트를 보완 재실행하는 루프는 설계(아키텍처 그림)에 있으나, 기본 코드에 해당 State가 없어 구현하지 못함
- 현장 도입·사업성 Agent는 실제 기업 1개 실행 결과를 원문과 대조했을 때 단계·수치 오류가 확인되어 프롬프트를 개선했고, 재실행 검증은 하지 못함
- 테스트의 LLM 응답은 대체 응답이라 실제 판단 품질을 보증하지 않음
- 출처 ID 검증은 출처의 존재를 확인하며, 출처가 주장과 일치하는지까지 보장하지는 않음

## Contributors

- 박주윤 : 기술·제품 검증 Agent (기술 문서 적재, CRAG, 환각 방지 장치)
- 배영환 : 현장 도입·사업성 Agent, 투자 판단 Agent (근거 기반 분석, 가중치 점수 산출·판단)
- 배은빈 : 분야·고객 분류 Agent (세부 분야·지불 고객·고객 문제 정의, 정보 부족 처리, 단위 테스트)
- 윤도균 : 스타트업 탐색 Agent, 보고서 생성 Agent (후보 검증, 5페이지 PDF 생성)
- 이경민 : 시장성·경쟁 Agent (시장 문서 적재, CRAG, 경쟁사 선정 고정, 검색 품질 평가·개선)
