"""Day 5: observability — request logging and optional Langfuse tracing.

RequestLogger appends one JSON object per line to DATA_DIR (gitignored).
Langfuse tracing activates ONLY when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are both set; otherwise every call is a no-op and
the agent runs exactly as before. Keys are never printed or logged.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


def data_dir() -> Path:
    """DATA_DIR, created on demand (resolves relative to backend/)."""
    d = Path(settings.DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


class RequestLogger:
    """Append-only JSONL log."""

    def __init__(self, filename: str = "requests.jsonl"):
        self.filename = filename

    def _path(self) -> Path:
        return data_dir() / self.filename

    def log(self, record: dict) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with open(self._path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @contextmanager
    def timed(self, endpoint: str):
        """Yield a dict for extra fields; logs endpoint + latency on exit."""
        extras: dict = {}
        start = time.perf_counter()
        try:
            yield extras
        finally:
            extras["endpoint"] = endpoint
            extras["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.log(extras)


request_logger = RequestLogger()
feedback_logger = RequestLogger(filename="feedback.jsonl")


class LangfuseTracer:
    """Optional Langfuse tracing.

    Instantiated once at import. Inert unless both LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY are set AND the langfuse package imports.
    Tracing can never break the app: any failure degrades to a no-op.
    """

    def __init__(self):
        self._client = None
        pk = settings.LANGFUSE_PUBLIC_KEY
        sk = settings.LANGFUSE_SECRET_KEY
        if pk and sk:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=pk, secret_key=sk, host=settings.LANGFUSE_HOST
                )
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(self, name: str, metadata: dict | None = None):
        """No-op context manager when tracing is disabled."""
        if self._client is None:
            yield None
            return
        trace = self._client.trace(name=name, metadata=metadata or {})
        try:
            yield trace
        finally:
            try:
                self._client.flush()
            except Exception:
                pass


tracer = LangfuseTracer()
