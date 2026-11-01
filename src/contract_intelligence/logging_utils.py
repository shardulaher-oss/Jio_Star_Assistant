"""Structured logging that never records contract or query text.

Log records carry event fields (pipeline, latency, counts, scores, a short query
hash) so behaviour can be monitored, while the text itself, which may be
confidential, stays out of log files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload |= {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "WARNING", json_format: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("contract_intelligence")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def query_id(query: str) -> str:
    """Short, non-reversible identifier for correlating log lines about one query."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
