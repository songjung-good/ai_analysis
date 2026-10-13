"""결정·사유 로그를 State가 아닌 외부 파일(JSONL)로 남긴다. trace_id로 State와 상관."""
import json
import time
from pathlib import Path

LOG_DIR = Path("outputs/logs")


def log_decision(trace_id, node, decision, reasons=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "trace_id": trace_id,
        "node": node,
        "decision": decision,
        "reasons": reasons,
        "ts": time.time(),
    }
    with (LOG_DIR / f"{trace_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
