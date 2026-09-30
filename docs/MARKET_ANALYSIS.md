# 시장성·경쟁 Agent 계약

`agents/market.py`가 읽는 입력과 `state["market_analysis"]`에 쓰는 출력을 정의합니다.
출력 타입은 `MarketAnalysisOutput`이며 반환 직전에 이 모델로 검증합니다.

## 입력

| State 키 | 필드 | 필수 | 용도 |
|---|---|---|---|
| `selected_startup` | `name` | O | 경쟁사 목록에서 자기 자신 제외 |
| `profile` | `subdomain` | O | 시장 규모 검색 질의, 스타트업맵 경쟁 분야 결정 |
| `profile` | `paying_customer` | | 수요 검색 질의 |
| `profile` | `customer_problem` | | 수요·경쟁 검색 질의 |
| `profile` | `industry` | | 스타트업맵 적용 산업 8개 중 하나. 있으면 `subdomain` 대응보다 우선 |
| `profile` | `tech_type` | | 스타트업맵 기술 유형 4개 중 하나. 있으면 `subdomain` 대응보다 우선 |

필수 필드가 없으면 추정하지 않고 `ValueError`를 냅니다. `"정보 부족"` 값은 없는 것으로 처리합니다.
`subdomain`, `paying_customer`, `customer_problem`은 `agents/profile.py`의 `CustomerProfile`이 채웁니다. `industry`, `tech_type`은 현재 profile에 없어 `subdomain`을 스타트업맵 분야로 대응시켜 씁니다.

| `subdomain` | 스타트업맵 분야 |
|---|---|
| 산업용·제조 로봇 | 제조·산업 / 로봇 |
| 물류 로봇 | 물류·유통 / 로봇, 자율주행 |
| 자율주행·모빌리티 | 모빌리티·교통 / 자율주행 |
| 서비스 로봇 | 서비스·생활 / 로봇 |
| 로봇 AI 소프트웨어 | (전 산업) / AI·SW 플랫폼 |
| 휴머노이드, 로봇 부품·하드웨어, 기타 | 대응 분야 없음 → LLM이 근거에서 경쟁사 선정 |

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
    "competitors": [Competitor],     # 최대 6개, 아래 경쟁사 선정 참고
    "competitor_landscape": {
        "segment": str | None,       # "제조·산업 로봇", 대응 분야가 없으면 None
        "rows": [{"industry", "tech_type", "companies": [str], "source_ids"}],
        "company_count": int,        # 같은 분야 국내 스타트업 수(경쟁 강도)
        "selected": [str],           # competitors로 상세 분석한 기업, 선정 순서
        "evidence_sentences": {name: [str]},  # 선정 기업마다 근거에서 찾은 문장
        "other_mentioned": [str],    # 선정 밖에서 근거에 등장한 경쟁사(이름만)
    },
    "differentiation": {"strengths": [Claim], "weaknesses": [Claim], "entry_barriers": [Claim]},
    "missing_info": [str],           # 근거가 없어 판단하지 못한 항목
    "evidence_level": "high" | "medium" | "low",
    "source_tiers": {source_id: 1 | 2 | 3},  # 인용한 출처의 등급, 아래 출처 정책 참고
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

## 경쟁사 선정

같은 입력이면 실행마다 같은 경쟁사가 나오도록 선정은 코드가 하고, LLM은 설명만 씁니다.

1. `subdomain`(또는 `industry`·`tech_type`)을 스타트업맵 분야로 대응시키고, 고정 질의로 스타트업맵의 해당 줄과 관련 본문을 검색합니다. 같은 DB면 결과가 같습니다.
2. 해당 줄의 기업 전체가 `competitor_landscape.rows`입니다. 평가 대상 기업은 제외합니다.
3. 본문에 이름이 나오는 기업을 먼저, 나머지는 스타트업맵 순서로 최대 6개를 `selected`로 고릅니다.
4. 코드가 기업마다 이름이 나오는 문장을 찾아 `evidence_sentences`에 넣고, LLM은 그 문장만 보고 `product`·`target_customer`·`differentiator`를 씁니다. 문장이 없는 기업은 LLM을 부르지 않고 `"근거 없음"`입니다.
5. 대응 분야가 없는 `subdomain`은 이전처럼 LLM이 근거에서 경쟁사를 고르고, 아래 검증을 거칩니다.

LLM 요약 문구는 실행마다 조금씩 달라질 수 있으므로, 사실 확인은 `evidence_sentences`를 기준으로 합니다.

## 코드가 보장하는 검증

LLM 출력을 그대로 믿지 않고 반환 전에 다음을 적용합니다. 결과는 `validation`에 남습니다.

| 검사 | 처리 | `validation` 키 |
|---|---|---|
| 존재하지 않는 `source_id` 인용 | 해당 id 제거, 유효한 인용이 없으면 항목 제거 | `dropped_uncited_items` |
| 수치가 인용한 1~2등급 근거에 없음 | 값과 단위가 함께 적힌 다른 1~2등급 근거로 인용 교정 | `recited_figures` |
| 수치가 어떤 1~2등급 근거에도 없음 | 수치 제거 | `unverified_figures` |
| 수치가 3등급 출처에만 있음 | 수치 제거 | `low_tier_figures` |
| 같은 지표·연도·지역·단위의 값이 1.5배 넘게 다름 | 등급이 높은 출처의 값만 남김 | `conflicting_figures` |
| 경쟁사명이 인용 근거에 없음 | 이름이 실린 다른 근거로 인용 교정 | `recited_competitors` |
| 경쟁사명이 어떤 근거에도 없음 | 경쟁사 제거 | `unverified_competitors` |
| 스타트업맵 산업이 `profile.industry`와 다름 | 경쟁사 제거 | `industry_mismatched_competitors` |
| 평가 대상 자신이 경쟁사에 포함 | 제거 | `excluded_self_as_competitor` |

수치 비교는 원문 표기 차이를 허용합니다. "132억 5천만"은 132.5억, "18억 5,800만"은 18.58억으로 읽고, LLM이 "542 (천 대)"를 542000대로 풀어 쓴 경우도 인용 근거와 대조합니다.

`evidence_level`은 LLM 판단이 아니라 CRAG 결과로 계산합니다. 시장·수요·경쟁 3개 주제 중 관련 근거가 2개 이상인 주제가 3개면 `high`, 2개면 `medium`, 그 이하면 `low`입니다.

## 출처 정책

| 등급 | 대상 | 수치 사용 |
|---|---|---|
| 1 | 적재한 PDF(`url` 없음), 공공기관(`.go.kr`, `.re.kr`, `.gov`), 국제기구·산업기관(IFR, OECD, 한국로봇산업진흥원 등) | O |
| 2 | 언론, 증권사·컨설팅, 데이터 제공사(Statista, MarketsandMarkets, IDC 등) | O |
| 3 | 그 외(기업 홈페이지, 시장조사 보고서 판매 사이트 등) | X, 서술·경쟁사 근거로만 사용 |
| 제외 | 블로그(tistory, 네이버 블로그, blogspot, brunch, medium 등) | 웹 결과에서 제거 |

웹 fallback은 1~2등급 도메인으로 먼저 검색하고, 결과가 3개 미만일 때만 일반 검색으로 넓힙니다. 도메인 목록은 `agents/market.py`의 `TIER1_DOMAINS`, `TIER2_DOMAINS`, `BLOCKED_DOMAINS`입니다. `tools/web.py`는 공용이라 건드리지 않았으며, 다른 Agent도 같은 기준이 필요하면 공용으로 옮기는 것을 제안합니다.

## 투자 판단 Agent 연결 (제안)

평가표의 `market`(25%)과 `competition`(10%) 점수는 투자 판단 Agent가 매깁니다. 다음 대응을 제안하며, 확정은 투자 판단 담당과 합의합니다.

| 점수 항목 | 주로 볼 필드 |
|---|---|
| `market` 시장 규모·성장·수요 | `market_size`, `growth`, `demand_drivers` |
| `competition` 진입장벽·차별성 | `competitors`, `differentiation`, `competitor_landscape.company_count` |

- `evidence_level == "low"`이면 해당 항목을 근거 부족으로 기록하고 임의 점수를 만들지 않습니다(`calculate_score`의 누락 항목 처리와 동일).
- 특허·기술 해자는 기술·제품 검증 Agent의 `technical_analysis`와 함께 봅니다.

## 보고서 생성 Agent 연결

- "3. 기술 및 시장 분석"의 시장 규모·성장 전망은 `market_size`, `growth`를 씁니다.
- 경쟁사 비교표는 `competitors`의 `name`, `type`, `product`, `differentiator` 4열을 권장합니다. 경쟁 강도는 `competitor_landscape.company_count`로 적습니다.
- 인용은 `source_ids`로 `references`와 연결합니다. 웹 출처는 `Evidence.url`, PDF 출처는 `Evidence.page`가 있습니다.

## 데이터 준비

원본 PDF는 저장소에 없습니다. `data/market/sources.json`에 적힌 파일명으로 `data/market/raw/`에 넣고 적재합니다.

```bash
PYTHONPATH=src python -m ai_investment.ingest market --dry-run   # 청크만 확인
PYTHONPATH=src python -m ai_investment.ingest market             # Chroma 적재
```

이미지로만 된 페이지는 `data/market/transcripts/`의 수동 전사본을 병합합니다. 전사본을 고치면 다시 적재합니다.

## 알려진 한계

- 검증은 수치가 근거에 적혀 있는지만 확인하고, 지표명(metric)을 올바르게 붙였는지는 확인하지 않습니다. 예: 산업용 로봇 전체 성장률을 협동로봇 성장률로 적는 경우.
- `gpt-5-nano`는 단위 환산이나 자릿수 실수("380억"을 "38억")를 자주 합니다. 검증이 이런 수치를 걸러 내므로 틀린 수치는 남지 않지만, 실행에 따라 남는 수치 개수가 달라집니다.
- 스타트업맵에 대응 분야가 없는 `subdomain`(휴머노이드 등)은 LLM이 경쟁사를 고르므로 실행마다 결과가 달라지고, 비어 있을 수도 있습니다. profile에 `industry`·`tech_type`이 생기면 해결됩니다.
- 웹 결과 대부분에 발행일이 없어 최신성으로 거르지 못합니다.
- 한 번 실행에 LLM 호출이 5~14회(주제별 관련성 평가·재작성·웹 평가, 최종 분석, 경쟁사 설명)이며 `gpt-5-nano`(`reasoning_effort=low`) 기준 30~50초가 걸립니다.
