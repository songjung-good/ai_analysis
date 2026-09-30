"""Run Agent 4 against live APIs and save evidence for manual review."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_investment.agents.business import run
from ai_investment.tools.web import web_search


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "TAVILY_API_KEY")
               if not os.getenv(name, "").strip()]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return 1

    # Example context only; funding eligibility and team are not verified here.
    state = {
        "selected_startup": {
            "name": "Agility Robotics", "product": "Digit",
            "funding_stage": "", "team": [],
        },
        "profile": {
            "subdomain": "물류·창고 로봇", "paying_customer": "물류·창고 운영 기업",
            "customer_problem": "반복적인 물류 작업의 인력 확보와 자동화가 어렵다.",
            "info_sufficiency": "예시 입력: 사실 검증 필요",
            "missing_info": ["투자 단계 및 팀 정보 미확인", "고객·제품 문맥은 실행 예시이며 별도 검증 필요"],
        },
    }
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": os.environ["OPENAI_MODEL"], "input": state, "searches": [],
    }

    def recorded_search(query, *, max_results):
        print(f"Search {len(record['searches']) + 1}/3", flush=True)
        evidence = web_search(query, max_results=max_results)
        record["searches"].append({
            "query": query, "results": [item.model_dump() for item in evidence],
        })
        return evidence

    try:
        result = run(state, search=recorded_search)
        record["result"] = {
            "business_analysis": result["business_analysis"],
            "references": [item.model_dump() for item in result["references"]],
        }
        record["status"] = "success"
    except Exception as exc:
        # Never print raw provider errors, which may contain request credentials.
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        print(f"Agent failed: {type(exc).__name__} (provider details hidden)")

    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / "outputs" / f"agility_business_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    if record["status"] == "success":
        print(f"Stage: {record['result']['business_analysis']['commercialization_stage']}")
        print(f"Cited references: {len(record['result']['references'])}")
        print(record["result"]["business_analysis"]["summary"])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
