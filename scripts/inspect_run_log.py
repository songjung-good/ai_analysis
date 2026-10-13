"""Read a run/attempt JSONL timeline without calling any external service."""
import argparse
import json
import time
from pathlib import Path


def timeline(path):
    operations = {}
    finished = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # A running process may still be writing the final event.
            raise
        operation_id = record.get("operation_id")
        event = record.get("event", "")
        if not operation_id:
            continue
        if event.endswith("_started"):
            operations[operation_id] = record
        elif event.endswith(("_finished", "_failed")):
            operations.pop(operation_id, None)
            finished.append(record)
    pending = [{**record, "status": "running", "elapsed_ms": (time.time() - record["ts"]) * 1000}
               for record in operations.values()]
    return finished, pending


def main():
    parser = argparse.ArgumentParser(description="실행·시도별 로그의 소요 시간과 오류 확인")
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--slowest", type=int, help="노드·전체 시간과 중복되지 않도록 내부 작업 중 느린 순서 N개")
    args = parser.parse_args()
    finished, pending = timeline(args.log_file)
    if args.slowest is not None:
        if args.slowest < 1:
            parser.error("--slowest must be at least 1")
        finished = sorted((record for record in finished if record.get("operation") not in {"run", "node"}),
                          key=lambda record: record.get("elapsed_ms", 0), reverse=True)[:args.slowest]
    for record in [*finished, *pending]:
        status = {"completed": "완료", "failed": "실패", "insufficient": "근거 부족",
                  "running": "진행 중"}.get(record.get("status"), record.get("status"))
        print(f"{status} | {record.get('elapsed_ms', 0) / 1000:.1f}초 | "
              f"{record.get('node')}/{record.get('operation')} | {record.get('target_name') or '-'} | "
              f"노드 시도 {record.get('node_attempt', '-')} / 호출 시도 {record.get('attempt', '-')} | "
              f"입력 {record.get('input_chars', '-')}자 / 검색 결과 {record.get('result_count', '-')}")
        if record.get("query"):
            print(f"  검색어: {record['query']}")
        if record.get("error"):
            print(f"  오류: {record.get('error_type', '')} {record['error']}")


if __name__ == "__main__":
    main()
