from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import NodeResult
from ..state import GraphState, Startup
from ..tools import tools_for


TOOLS = tools_for("customer_profile")

Subdomain = Literal[
    "휴머노이드",
    "산업용·제조 로봇",
    "물류 로봇",
    "자율주행·모빌리티",
    "서비스 로봇",
    "로봇 AI 소프트웨어",
    "로봇 부품·하드웨어",
    "기타",
]

SYSTEM_PROMPT = """너는 Physical AI·Robotics 스타트업 투자 심사팀의 분류 담당이다.
주어진 스타트업 정보만 근거로 아래 세 가지를 정의한다.

1. subdomain: 세부 분야. 정해진 목록에서 가장 가까운 하나만 고른다.
2. paying_customer: 제품에 실제로 비용을 지불하는 고객 (예: 제조 공장, 물류센터).
   사용자와 지불자가 다르면 지불자를 적는다.
3. customer_problem: 그 고객이 해결하려는 문제를 한두 문장으로 적는다.

규칙:
- 제공된 정보에 없는 내용은 추측하지 않는다. 근거가 없으면 해당 필드는 "정보 부족"으로 적는다.
- info_sufficiency는 세 항목 모두 제공된 정보로 판단했으면 "충분", 하나라도 "정보 부족"이면 "부족"으로 적는다.
- missing_info에는 부족했던 정보의 종류를 적는다 (예: "주요 고객", "제품 설명").
- 한국어로 답한다."""


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subdomain: Subdomain
    paying_customer: str = Field(min_length=1)
    customer_problem: str = Field(min_length=1)
    info_sufficiency: Literal["충분", "부족"]
    missing_info: list[str] = Field(default_factory=list)


def _build_classifier():
    """Create the structured-output LLM. Imported lazily so tests need no API key."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Configure it in the environment or .env.")
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        CustomerProfile
    )


def _describe(startup: Startup) -> str:
    team = ", ".join(startup.get("team") or []) or "정보 없음"
    return (
        f"기업명: {startup['name']}\n"
        f"제품: {startup.get('product') or '정보 없음'}\n"
        f"투자 단계: {startup.get('funding_stage') or '정보 없음'}\n"
        f"팀 구성: {team}"
    )


def classify(startup: Startup, classifier) -> dict[str, object]:
    """Classify one startup. `classifier.invoke(messages)` returns CustomerProfile."""
    if not (startup.get("name") or "").strip():
        raise ValueError("selected_startup must have a name")

    # Without a product description the LLM can only guess, so skip the call.
    if not (startup.get("product") or "").strip():
        return CustomerProfile(
            subdomain="기타",
            paying_customer="정보 부족",
            customer_problem="정보 부족",
            info_sufficiency="부족",
            missing_info=["제품 설명"],
        ).model_dump()

    result = classifier.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", _describe(startup)),
        ]
    )
    if not isinstance(result, CustomerProfile):
        result = CustomerProfile.model_validate(result)
    return result.model_dump()


def run(state: GraphState) -> NodeResult:
    """Classify subdomain, paying customer, and customer problem."""
    startup = state.get("selected_startup")
    if not startup:
        raise ValueError("selected_startup is required before customer_profile")

    classifier = _build_classifier() if (startup.get("product") or "").strip() else None
    profile = classify(startup, classifier)
    # This agent uses no external source, so it adds no references.
    return {"profile": profile, "references": []}
