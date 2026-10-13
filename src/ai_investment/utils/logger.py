"""결정·사유 로그를 State가 아닌 외부 파일(JSONL)로 남긴다. trace_id로 State와 상관."""
import json
import logging
import time
from pathlib import Path

LOG_DIR = Path("outputs/logs")


class TraceFormatter(logging.Formatter):
    def __init__(self, trace_id):
        super().__init__()
        self.trace_id = trace_id

    def format(self, record):
        rec = {
            "trace_id": getattr(record, "trace_id", None) or self.trace_id,
            "node": getattr(record, "node", record.name),
            "message": record.getMessage(), "ts": record.created,
        }
        for key in ("event", "view", "target_name", "attempt", "query", "rework_hints",
                    "result_count", "error", "decision", "reason", "target_views",
                    "retry_count", "delay_seconds"):
            if key in record.__dict__:
                rec[key] = record.__dict__[key]
        return json.dumps(rec, ensure_ascii=False, default=str)


def setup_logging(trace_id):
    """Attach structured business/evaluator logging to this run's decision log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / f"{trace_id}.jsonl", encoding="utf-8")
    handler.setFormatter(TraceFormatter(trace_id))
    logger = logging.getLogger("ai_investment")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler


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
