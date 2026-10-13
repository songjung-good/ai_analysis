"""Multi-Agent investment analysis pipeline entrypoint."""

import argparse
import importlib
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))
# LangSmith 등 환경 변수는 그래프 실행 전에 읽어야 한다
load_dotenv()

from ai_investment.agents import (  # noqa: E402
    business_analysis,
    customer_profile,
    investment_decision,
    market_analysis,
    report_generation,
    startup_discovery,
    technical_analysis,
)
from ai_investment.agents.market import run_view as market_view  # noqa: E402
from ai_investment.graph import AgentNodes, build_graph  # noqa: E402
from ai_investment.state import create_initial_state  # noqa: E402

def stakeholder_analysis(state):
    """business 에이전트를 이해관계자 관점으로 호출한다."""
    return business_analysis(state, view="stakeholder")


def domain_analysis(state):
    """business 에이전트를 도메인 적용 관점으로 호출한다."""
    return business_analysis(state, view="domain")


# supervisor 구조에 필요한 노드의 위치. module이 None이면 이 파일에 정의된 함수를 쓴다.
# (technical/market은 기존 technical_analysis/market_analysis 함수를 그대로 쓴다)
SUPERVISOR_NODE_SPEC = {
    "supervisor": ("ai_investment.orchestration.supervisor", "supervisor"),
    "evaluator": ("ai_investment.orchestration.evaluator", "run"),
    "stakeholder": (None, "stakeholder_analysis"),
    "domain": (None, "domain_analysis"),
}


def load_supervisor_nodes():
    """supervisor 구조에 필요한 노드를 불러온다. (불러온 노드, 없는 노드 목록)를 돌려준다."""
    loaded, missing = {}, []
    for field, (module, attr) in SUPERVISOR_NODE_SPEC.items():
        label = f"{module or 'app'}.{attr}"
        try:
            source = globals() if module is None else vars(importlib.import_module(module))
            loaded[field] = source[attr]
        except (ImportError, KeyError):
            missing.append(f"{field} ({label})")
    return loaded, missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-agent AI investment analysis system."
    )
    parser.add_argument(
        "--domain",
        default="Physical AI / Robotics",
        help="Target startup domain (default: Physical AI / Robotics)",
    )
    parser.add_argument(
        "--region",
        default="대한민국",
        help="Target region for discovery (default: 대한민국)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max candidate startups to discover (default: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max evaluation iterations before final report (default: 3)",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Direct target startup name to analyze (e.g. 플라잎, 베어로보틱스, Agility Robotics)",
    )
    parser.add_argument(
        "--product",
        default=None,
        help="--company와 함께 쓰는 제품 설명. 확인된 내용만 입력한다 (없으면 비워 둠)",
    )
    parser.add_argument(
        "--funding-stage",
        default=None,
        help="--company와 함께 쓰는 투자 단계. 확인된 내용만 입력한다 (없으면 비워 둠)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "supervisor", "legacy"],
        default="auto",
        help="auto: supervisor 노드가 모두 있으면 supervisor 구조, 없으면 기존 구조 (default: auto)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Supervisor 호출 상한 (default: 20)",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit (default: 100)",
    )
    parser.add_argument(
        "--check-nodes",
        action="store_true",
        help="supervisor 구조에 필요한 노드가 준비됐는지만 확인하고 종료 (API 호출 없음)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    supervisor_nodes, missing = load_supervisor_nodes()
    if args.check_nodes:
        print("supervisor 구조 노드 점검")
        for field in SUPERVISOR_NODE_SPEC:
            print(f"  {'OK     ' if field in supervisor_nodes else '누락   '} {field}")
        print("준비 완료" if not missing else f"누락: {', '.join(missing)}")
        return

    # ponytail: allow direct target company for deterministic evaluation
    if args.company:
        company_name = args.company.strip()

        def targeted_discovery(state):
            print(f"🎯 지정된 기업 '{company_name}' 분석 파이프라인 진입")
            # 확인되지 않은 제품·투자 단계를 임의로 채우지 않는다. 주어진 값만 쓴다.
            startup = {"name": company_name, "team": []}
            if args.product:
                startup["product"] = args.product.strip()
            if args.funding_stage:
                startup["funding_stage"] = args.funding_stage.strip()
            if not startup.get("product"):
                print("⚠ --product가 없어 제품 설명 없이 진행합니다 (프로필은 '정보 부족'으로 처리)")
            return {
                "candidates": [startup],
                "current_idx": 0,
                "selected_startup": startup,
                "references": [],
            }

        discovery_node = targeted_discovery
    else:
        discovery_node = startup_discovery

    # supervisor 구조 사용 여부 결정
    if args.mode == "supervisor" and missing:
        raise SystemExit(f"supervisor 구조에 필요한 노드가 없습니다: {', '.join(missing)}")
    use_supervisor = args.mode != "legacy" and not missing
    if args.mode == "auto" and missing:
        print(f"⚠ supervisor 노드 누락({', '.join(missing)}) → 기존 구조로 실행합니다")

    # ponytail: default nodes assembly from canonical agent implementations
    nodes = AgentNodes(
        startup_discovery=discovery_node,
        customer_profile=customer_profile,
        technical_analysis=technical_analysis,
        business_analysis=business_analysis,
        # supervisor 구조는 공통 관점 결과(analyses)를 돌려주는 run_view를 써야 한다
        market_analysis=market_view if use_supervisor else market_analysis,
        investment_decision=investment_decision,
        report_generation=report_generation,
        **(supervisor_nodes if use_supervisor else {}),
    )

    trace_id = f"{datetime.now():%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    print("=" * 60)
    print("AI 스타트업 투자 평가 Multi-Agent 시스템 시작")
    print(f"도메인: {args.domain} | 지역: {args.region} | 후보 한도: {args.limit}")
    print(f"구조: {'supervisor' if use_supervisor else 'legacy'} | trace_id: {trace_id}")
    print(
        "LangSmith: "
        + (f"켜짐 (project={os.getenv('LANGSMITH_PROJECT', '-')})" if tracing else "꺼짐")
    )
    print("=" * 60)

    graph = build_graph(nodes)
    initial_state = create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit},
        max_iterations=args.max_iterations,
        max_steps=args.max_steps,
        trace_id=trace_id,
    )
    config = {
        "run_name": f"supervisor-{trace_id}",
        "metadata": {
            "trace_id": trace_id,
            "domain": args.domain,
            "company": args.company,
            "mode": "supervisor" if use_supervisor else "legacy",
        },
        "tags": ["agent-supervisor"],
        "recursion_limit": args.recursion_limit,
    }

    final_state = graph.invoke(initial_state, config=config)

    print("\n" + "=" * 60)
    print("분석 완료")
    if "report" in final_state:
        print(f"📄 최종 보고서 생성 경로: {final_state['report']}")
    if final_state.get("decision"):
        print(
            f"📊 최종 투자 판단: {final_state.get('decision')} "
            f"(점수: {final_state.get('investment_score')})"
        )
    if use_supervisor:
        verdict = final_state.get("eval_result") or {}
        print(
            f"🔁 보고서 평가: 통과={verdict.get('passed')} | "
            f"미달 횟수={final_state.get('report_attempts')} | "
            f"supervisor 호출={final_state.get('step_count')} | "
            f"재작업={final_state.get('rework_count')}"
        )
    print(f"🧭 trace_id: {trace_id}  (결정 로그: outputs/logs/{trace_id}.jsonl)")
    print("=" * 60)


if __name__ == "__main__":
    main()
