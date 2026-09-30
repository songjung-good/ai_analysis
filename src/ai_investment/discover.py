"""Standalone discovery command: python -m ai_investment.discover."""

import argparse
import json

from .agents.discovery import run
from .state import create_initial_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover eligible Physical AI/Robotics startups")
    parser.add_argument("--domain", default="Physical AI/Robotics")
    parser.add_argument("--region", default=None)
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=5)
    args = parser.parse_args()
    result = run(create_initial_state(
        domain=args.domain,
        criteria={"region": args.region, "candidate_limit": args.limit},
        max_iterations=args.limit,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda item: item.model_dump()))


if __name__ == "__main__":
    main()
