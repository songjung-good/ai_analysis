"""결정 로그 진입점.

배은빈 담당 utils/logger.py의 log_decision(trace_id, node, decision, reasons)을 사용한다.
해당 모듈이 아직 머지되지 않은 브랜치에서도 동작하도록, 없으면 표준 logging으로 같은 내용을 남긴다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from ..utils.logger import log_decision as _log_decision
except ImportError:  # utils/logger.py 머지 전
    _logger = logging.getLogger("ai_investment.decision")

    def _log_decision(trace_id: str | None, node: str, decision: Any, reasons: Any) -> None:
        _logger.info(json.dumps(
            {"trace_id": trace_id, "node": node, "decision": decision, "reasons": reasons},
            ensure_ascii=False, default=str,
        ))


def log_decision(trace_id: str | None, node: str, decision: Any, reasons: Any) -> None:
    _log_decision(trace_id, node, decision, reasons)
