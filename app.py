"""Multi-Agent investment analysis pipeline entrypoint."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_investment.agents import (
    business_analysis,
    customer_profile,
    investment_decision,
    market_analysis,
    report_generation,
    startup_discovery,
    technical_analysis,
)
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.state import create_initial_state


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
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # ponytail: allow direct target company for deterministic evaluation
    if args.company:
        company_name = args.company.strip()

        def targeted_discovery(state):
            print(f"🎯 지정된 기업 '{company_name}' 분석 파이프라인 진입")
            startup = {
                "name": company_name,
                "product": f"{company_name} 로봇 AI 솔루션",
                "funding_stage": "Series A",
                "team": [],
            }
            return {
                "candidates": [startup],
                "current_idx": 0,
                "selected_startup": startup,
                "references": [],
            }

        discovery_node = targeted_discovery
    else:
        discovery_node = startup_discovery

    # ponytail: default nodes assembly from canonical agent implementations
    nodes = AgentNodes(
        startup_discovery=discovery_node,
        customer_profile=customer_profile,
        technical_analysis=technical_analysis,
        business_analysis=business_analysis,
        market_analysis=market_analysis,
        investment_decision=investment_decision,
        report_generation=report_generation,
    )

    print("=" * 60)
    print("AI 스타트업 투자 평가 Multi-Agent 시스템 시작")
    print(f"도메인: {args.domain} | 지역: {args.region} | 후보 한도: {args.limit}")
    print("=" * 60)

    graph = build_graph(nodes)
    initial_state = create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit},
        max_iterations=args.max_iterations,
    )

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("분석 완료")
    if "report" in final_state:
        print(f"📄 최종 보고서 생성 경로: {final_state['report']}")
    if "decision" in final_state:
        print(
            f"📊 최종 투자 판단: {final_state.get('decision')} "
            f"(점수: {final_state.get('investment_score')})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
