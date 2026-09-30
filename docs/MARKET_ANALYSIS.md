# 시장성·경쟁 Agent 계약

`agents/market.py`가 읽는 입력과 `state["market_analysis"]`에 쓰는 출력을 정의합니다.
출력 타입은 `MarketAnalysisOutput`이며 반환 직전에 이 모델로 검증합니다.

## 입력

| State 키 | 필드 | 필수 | 용도 |
|---|---|---|---|
| `selected_startup` | `name` | O | 경쟁사 목록에서 자기 자신 제외 |
| `profile` | `subdomain` | O | 시장 규모 검색 질의 |
| `profile` | `paying_customer` | | 수요 검색 질의 |
| `profile` | `customer_problem` | | 수요·경쟁 검색 질의 |
| `profile` | `industry` | | 스타트업맵 적용 산업 8개 중 하나. 경쟁사 검색·산업 불일치 제거 |
| `profile` | `tech_type` | | 스타트업맵 기술 유형 4개 중 하나. 경쟁사 검색 |

필수 필드가 없으면 추정하지 않고 `ValueError`를 냅니다. `"정보 부족"` 값은 없는 것으로 처리합니다.
`subdomain`, `paying_customer`, `customer_problem`은 `agents/profile.py`의 `CustomerProfile`이 채웁니다. `industry`, `tech_type`은 현재 profile에 없어 경쟁사 산업 검증이 꺼진 상태이며, 추가하면 바로 적용됩니다.

`industry` 값: 제조·산업, 물류·유통, 모빌리티·교통, 의료·헬스케어, 건설·인프라, 농업·식품, 서비스·생활, 국방·안보
`tech_type` 값: 로봇, 자율주행, 드론·UAM, AI·SW 플랫폼

## 출력

```python
market_analysis = {
    "summary": str,                  # 3문장 이내
    "subdomain": str,
    "market_size": [MarketFigure],   # 최대 4개, 지표별 최신 실측값
    "growth": [MarketFigure],        # 최대 4개, 전망치는 year에 "(전망)"
    "demand_drivers": [Claim],
    "competitors": [Competitor],     # 최대 6개
    "differentiation": {"strengths": [Claim], "weaknesses": [Claim], "entry_barriers": [Claim]},
    "missing_info": [str],           # 근거가 없어 판단하지 못한 항목
    "evidence_level": "high" | "medium" | "low",
    "validation": {...},             # 코드가 제거·교정한 항목
    "retrieval": {"market" | "demand" | "competition": {"queries", "used_web", "evidence_count"}},
}

MarketFigure = {"metric", "value": float, "unit", "year", "region", "source_ids"}
Claim = {"claim", "source_ids"}
Competitor = {"name", "type": "국내 스타트업" | "해외 스타트업" | "대기업·상장사",
              "product", "target_customer", "differentiator", "source_ids"}
```

- 모든 `source_ids`는 같은 실행에서 `references`에 추가된 `Evidence.source_id`입니다.
- `references`에는 출력에서 실제로 인용한 Evidence만 들어갑니다.
- 스타트업맵은 기업명만 담고 있어 다른 근거가 없으면 `product`, `differentiator`가 `"근거 없음"`입니다.

## 코드가 보장하는 검증

LLM 출력을 그대로 믿지 않고 반환 전에 다음을 적용합니다. 결과는 `validation`에 남습니다.

| 검사 | 처리 | `validation` 키 |
|---|---|---|
| 존재하지 않는 `source_id` 인용 | 해당 id 제거, 유효한 인용이 없으면 항목 제거 | `dropped_uncited_items` |
| 수치가 인용 근거 본문에 없음 | 수치 제거 | `unverified_figures` |
| 경쟁사명이 인용 근거에 없음 | 이름이 실린 다른 근거로 인용 교정 | `recited_competitors` |
| 경쟁사명이 어떤 근거에도 없음 | 경쟁사 제거 | `unverified_competitors` |
| 스타트업맵 산업이 `profile.industry`와 다름 | 경쟁사 제거 | `industry_mismatched_competitors` |
| 평가 대상 자신이 경쟁사에 포함 | 제거 | `excluded_self_as_competitor` |

`evidence_level`은 LLM 판단이 아니라 CRAG 결과로 계산합니다. 시장·수요·경쟁 3개 주제 중 관련 근거가 2개 이상인 주제가 3개면 `high`, 2개면 `medium`, 그 이하면 `low`입니다.

## 투자 판단 Agent 연결 (제안)

평가표의 `market`(25%)과 `competition`(10%) 점수는 투자 판단 Agent가 매깁니다. 다음 대응을 제안하며, 확정은 투자 판단 담당과 합의합니다.

| 점수 항목 | 주로 볼 필드 |
|---|---|
| `market` 시장 규모·성장·수요 | `market_size`, `growth`, `demand_drivers` |
| `competition` 진입장벽·차별성 | `competitors`, `differentiation` |

- `evidence_level == "low"`이면 해당 항목을 근거 부족으로 기록하고 임의 점수를 만들지 않습니다(`calculate_score`의 누락 항목 처리와 동일).
- 특허·기술 해자는 기술·제품 검증 Agent의 `technical_analysis`와 함께 봅니다.

## 보고서 생성 Agent 연결

- "3. 기술 및 시장 분석"의 시장 규모·성장 전망은 `market_size`, `growth`를 씁니다.
- 경쟁사 비교표는 `competitors`의 `name`, `type`, `product`, `differentiator` 4열을 권장합니다.
- 인용은 `source_ids`로 `references`와 연결합니다. 웹 출처는 `Evidence.url`, PDF 출처는 `Evidence.page`가 있습니다.

## 데이터 준비

원본 PDF는 저장소에 없습니다. `data/market/sources.json`에 적힌 파일명으로 `data/market/raw/`에 넣고 적재합니다.

```bash
PYTHONPATH=src python -m ai_investment.ingest market --dry-run   # 청크만 확인
PYTHONPATH=src python -m ai_investment.ingest market             # Chroma 적재
```

이미지로만 된 페이지는 `data/market/transcripts/`의 수동 전사본을 병합합니다. 전사본을 고치면 다시 적재합니다.

## 알려진 한계

- 웹 fallback은 출처 등급을 가리지 않습니다. 오래된 블로그의 수치가 들어올 수 있으므로 `Evidence.url`과 `published_at`을 함께 확인합니다.
- 검증은 수치가 근거에 적혀 있는지만 확인하고, 수치의 의미를 올바르게 옮겼는지는 확인하지 않습니다.
- 한 번 실행에 LLM 호출이 4~13회(주제별 관련성 평가·재작성·웹 평가 + 최종 분석)이며 `gpt-5-nano` 기준 2~3분이 걸립니다.
