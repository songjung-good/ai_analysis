"""Standalone discovery command: python -m ai_investment.discover."""

import argparse
import json
import logging
from pathlib import Path

from .agents.discovery import DiscoveryCriteria, run
from .state import create_initial_state
from .contracts import guard_node
from .utils.logger import setup_logging


def load_candidate_names(path: str | Path) -> list[str]:
    """Load search seeds, never pre-verified startup facts."""
    names = DiscoveryCriteria(candidate_names=json.loads(Path(path).read_text(encoding="utf-8"))).candidate_names
    if not names:
        raise ValueError("candidate list must not be empty")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover eligible Physical AI/Robotics startups")
    parser.add_argument("--domain", default="Physical AI/Robotics")
    parser.add_argument("--region", default=None)
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--candidate-list", type=Path, help="JSON array of company names to verify")
    args = parser.parse_args()
    try:
        names = load_candidate_names(args.candidate_list) if args.candidate_list else []
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    state = create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit, "candidate_names": names},
        max_iterations=args.limit,
    )
    handler = setup_logging(state["trace_id"])
    try:
        result = guard_node("startup_discovery", run)(state)
    finally:
        logging.getLogger("ai_investment").removeHandler(handler)
        handler.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda item: item.model_dump()))


if __name__ == "__main__":
    main()
