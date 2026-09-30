"""AI 스타트업 투자 평가 Multi-Agent 시스템 진입점."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# ponytail: minimal runner wiring all 7 agents without complex CLI flags. Upgrade path: add rich TUI or streaming UI.
load_dotenv()

# Add src to sys.path if not installed as package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 스타트업 투자 평가 Multi-Agent 시스템")
    parser.add_argument("--domain", default="Physical AI/Robotics", help="탐색 대상 도메인")
    parser.add_argument("--region", default=None, help="기업 소재 지역 (예: 대한민국)")
    parser.add_argument("--limit", type=int, default=3, help="탐색 및 평가 후보 상한 (1~10)")
    parser.add_argument("--max-iterations", type=int, default=3, help="최대 평가 반복 횟수")
    args = parser.parse_args()

    nodes = AgentNodes(
        startup_discovery=startup_discovery,
        customer_profile=customer_profile,
        technical_analysis=technical_analysis,
        business_analysis=business_analysis,
        market_analysis=market_analysis,
        investment_decision=investment_decision,
        report_generation=report_generation,
    )
    graph = build_graph(nodes)

    initial_state = create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit},
        max_iterations=args.max_iterations,
    )

    print(f"[*] AI 스타트업 투자 평가 파이프라인 시작: domain={args.domain}, region={args.region}, limit={args.limit}")
    result = graph.invoke(initial_state)

    print("\n[+] 평가 완료!")
    if "report" in result and result["report"]:
        print(f"[+] 최종 보고서 경로: {result['report']}")
    if "evaluations" in result:
        print(f"[+] 평가 이력: {len(result['evaluations'])}개 기업 검토 완료")
        for idx, ev in enumerate(result["evaluations"], 1):
            name = ev.get("startup_name", "미상")
            score = ev.get("investment_score")
            decision = ev.get("decision", "hold")
            print(f"  {idx}. {name}: {decision.upper()} (점수: {score})")


if __name__ == "__main__":
    main()
