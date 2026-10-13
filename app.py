"""Multi-Agent investment analysis pipeline entrypoint."""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_investment.agents import (
    business_analysis,
    customer_profile,
    investment_decision,
    report_generation,
    startup_discovery,
    technical_analysis,
)
from ai_investment.graph import AgentNodes, build_graph
from ai_investment.discover import load_candidate_names
from ai_investment.agents.market import run_view as market_view
from ai_investment.agents.report import finalize_report
from ai_investment.orchestration.evaluator import run as evaluate_report
from ai_investment.orchestration.supervisor import supervisor
from ai_investment.state import create_initial_state
from ai_investment.utils.logger import setup_logging, monitor


def supervised_decision(state):
    """Adapt business views and replace this candidate's score after report research."""
    from langgraph.types import Overwrite

    business = {
        view: result for view, result in (state.get("analyses") or {}).items()
        if view in {"stakeholder", "domain"}
    }
    update = investment_decision({**state, "business_analysis": business})
    history = state.get("evaluations") or []
    if state.get("report_draft") is not None and history and update.get("evaluations"):
        if history[-1]["startup_name"] == state["selected_startup"]["name"]:
            update = {**update, "evaluations": Overwrite([*history[:-1], *update["evaluations"]])}
    return update


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
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--company",
        default=None,
        help="Direct target startup name to analyze (e.g. 플라잎, 베어로보틱스, Agility Robotics)",
    )
    targets.add_argument("--candidate-list", type=Path, help="검증할 기업 이름 목록(JSON 파일)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max supervisor calls (default: 20)")
    args = parser.parse_args()
    try:
        args.candidate_names = load_candidate_names(args.candidate_list) if args.candidate_list else []
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.company is not None:
        args.company = args.company.strip()
        if not args.company:
            parser.error("--company must not be empty")
    return args


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()

    # A company name supplies no verified product or funding-stage facts.
    if args.company:
        company_name = args.company.strip()

        def targeted_discovery(state):
            print(f"🎯 지정된 기업 '{company_name}' 분석 파이프라인 진입")
            startup = {"name": company_name}
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
        market_analysis=market_view,
        investment_decision=supervised_decision,
        report_generation=report_generation,
        supervisor=supervisor,
        stakeholder=partial(business_analysis, view="stakeholder"),
        domain=partial(business_analysis, view="domain"),
        evaluator=evaluate_report,
        final_report=finalize_report,
    )

    print("=" * 60)
    print("AI 스타트업 투자 평가 Supervisor 시스템 시작")
    print(f"도메인: {args.domain} | 지역: {args.region} | 후보 한도: {args.limit}")
    print("=" * 60)

    graph = build_graph(nodes)
    initial_state = create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit,
                  "candidate_names": args.candidate_names},
        max_iterations=args.max_iterations,
        max_steps=args.max_steps,
    )

    handler = setup_logging(initial_state["trace_id"])
    print(f"실행 ID: {initial_state['trace_id']}")
    print(f"실행 로그: {handler.baseFilename}")
    print(f"시도별 로그: {Path(handler.baseFilename).with_suffix('') / 'attempts'}")
    try:
        with monitor("run", label="전체 투자 분석 실행", trace_id=initial_state["trace_id"]):
            final_state = graph.invoke(initial_state, config={
                "metadata": {"trace_id": initial_state["trace_id"]},
                "run_name": f"supervisor-{initial_state['trace_id']}",
                "recursion_limit": max(100, 2 * args.max_steps + 4 * args.max_iterations + 20),
            })
    finally:
        logging.getLogger("ai_investment").removeHandler(handler)
        handler.close()

    print("\n" + "=" * 60)
    print("분석 완료")
    if "report" in final_state:
        print(f"📄 최종 보고서 생성 경로: {final_state['report']}")
    verdict = final_state.get("eval_result") or {}
    print(f"보고서 품질: {'통과' if verdict.get('passed') else '미통과'}")
    if "decision" in final_state:
        print(
            f"📊 최종 투자 판단: {final_state.get('decision')} "
            f"(점수: {final_state.get('investment_score')})"
        )
    print("=" * 60)
    return final_state


if __name__ == "__main__":
    main()
