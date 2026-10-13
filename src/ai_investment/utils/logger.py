"""결정·사유 로그를 State가 아닌 외부 파일(JSONL)로 남긴다. trace_id로 State와 상관."""
import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LOG_DIR = Path("outputs/logs")
_context = ContextVar("investment_log_context", default={})


class TraceFormatter(logging.Formatter):
    def __init__(self, trace_id):
        super().__init__()
        self.trace_id = trace_id

    def format(self, record):
        rec = {
            **_context.get(),
            "trace_id": getattr(record, "trace_id", None) or self.trace_id,
            "node": getattr(record, "node", record.name),
            "message": record.getMessage(), "ts": record.created,
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        }
        for key in ("event", "view", "target_name", "attempt", "query", "rework_hints",
                    "result_count", "error", "decision", "reason", "target_views",
                    "retry_count", "delay_seconds", "attempt_id", "node_attempt",
                    "operation_id", "parent_operation_id", "operation", "status",
                    "elapsed_ms", "error_type", "traceback", "model", "schema",
                    "input_chars", "evidence_count", "evidence_chars", "source_ids",
                    "source_urls", "timeout_seconds", "sdk_max_retries", "missing_count",
                    "candidate_count", "query_index", "query_count", "artifact", "reasons",
                    "checks", "missing_fields"):
            if key in record.__dict__:
                rec[key] = record.__dict__[key]
        return json.dumps(rec, ensure_ascii=False, default=str)


class RunLogHandler(logging.FileHandler):
    """Keep the run timeline and a separate file for each node invocation."""

    def emit(self, record):
        try:
            super().emit(record)
            context = _context.get()
            if context.get("attempt_id"):
                directory = Path(self.baseFilename).with_suffix("") / "attempts"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{context['node']}-{context['attempt_id']}.jsonl"
                # ponytail: append per event; keep files open if event volume becomes large.
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


@contextmanager
def monitor(operation, *, label=None, logger_name="ai_investment.monitor", **fields):
    """Measure one operation/attempt without changing retries or error propagation."""
    parent = _context.get()
    details = {**parent, **fields, "operation": operation, "operation_id": uuid4().hex}
    if parent.get("operation_id"):
        details["parent_operation_id"] = parent["operation_id"]
    token = _context.set(details)
    logger = logging.getLogger(logger_name)
    caption = label or operation
    target = details.get("target_name")
    caption += f" [대상: {target}]" if target else ""
    if details.get("query_index"):
        caption += f" [검색 {details['query_index']}/{details['query_count']}]"
    attempt = details.get("node_attempt", details.get("attempt", 0) + 1) if operation == "search" else details.get("attempt", 1)
    started = time.perf_counter()
    metrics = {}
    logger.info("%s 시작", caption, extra={**details, "event": f"{operation}_started", "status": "running"})
    print(f"▶ {caption} (시도 {attempt})", flush=True)
    try:
        yield metrics
    except BaseException as exc:
        metrics.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                       traceback=[{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(exc.__traceback__)])
        raise
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        status = metrics.get("status", "completed")
        event = f"{operation}_failed" if status == "failed" else f"{operation}_finished"
        logger.info("%s %s (%.1f초)", caption, status, elapsed / 1000,
                    extra={**details, **metrics, "event": event, "status": status,
                           "elapsed_ms": round(elapsed, 2)})
        mark = "✖" if status == "failed" else "△" if status == "insufficient" else "✔"
        suffix = f"; {metrics['error']}" if metrics.get("error") else ""
        if "result_count" in metrics and operation != "node":
            suffix += f"; 결과 {metrics['result_count']}개"
        if "candidate_count" in metrics:
            suffix += f"; 후보 {metrics['candidate_count']}개"
        status_label = {"completed": "완료", "failed": "실패", "insufficient": "완료·근거 부족"}.get(status, status)
        print(f"{mark} {caption} {status_label} ({elapsed / 1000:.1f}초){suffix}", flush=True)
        _context.reset(token)


@contextmanager
def monitor_node(node, state, label):
    startup = state.get("selected_startup") or {}
    view = {"technical_analysis": "technical", "market_analysis": "market"}.get(node, node)
    count = (state.get("rework_count") or {}).get(view, 0)
    if node in {"report_generation", "evaluator", "investment_decision"}:
        count = state.get("report_attempts", 0)
    elif node == "supervisor":
        count = state.get("step_count", 0)
    token = _context.set({**_context.get(), "trace_id": state.get("trace_id"), "node": node,
                          "target_name": startup.get("name") if isinstance(startup, dict) else None,
                          "node_attempt": count + 1,
                          "attempt": count + 1, "attempt_id": uuid4().hex})
    try:
        with monitor("node", label=label) as metrics:
            yield metrics
    finally:
        _context.reset(token)


def invoke_logged(runnable, payload, *, label, model=None, attempt=1, **fields):
    """Log LLM latency/input size; do not persist prompts or credentials."""
    fields.update(input_chars=len(payload) if isinstance(payload, str) else
                  len(json.dumps(payload, ensure_ascii=False, default=str)),
                  model=getattr(model, "model_name", None) or os.getenv("OPENAI_MODEL"))
    if model is not None:
        fields.update(timeout_seconds=getattr(model, "request_timeout", None),
                      sdk_max_retries=getattr(model, "max_retries", None))
    with monitor("llm", label=label, attempt=attempt, **fields):
        return runnable.invoke(payload)


def setup_logging(trace_id):
    """Attach structured business/evaluator logging to this run's decision log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RunLogHandler(LOG_DIR / f"{trace_id}.jsonl", encoding="utf-8")
    handler.setFormatter(TraceFormatter(trace_id))
    logger = logging.getLogger("ai_investment")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler


def log_decision(trace_id, node, decision, reasons=None):
    trace_id = trace_id or _context.get().get("trace_id")
    rec = {
        **_context.get(),
        "trace_id": trace_id,
        "node": node,
        "decision": decision,
        "reasons": reasons,
        "ts": time.time(),
    }
    logger = logging.getLogger("ai_investment.decisions")
    if any(isinstance(handler, RunLogHandler) and handler.formatter.trace_id == trace_id
           for handler in logging.getLogger("ai_investment").handlers):
        logger.info("라우팅 판단: %s", decision, extra={**rec, "event": "routing_decision"})
        return
    if not trace_id:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"{trace_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
