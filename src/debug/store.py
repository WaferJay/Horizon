"""Persistent, opt-in debug artifacts for a Horizon run."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .._file_utils import _atomic_write_text
from .context import current_debug_context
from .redaction import (
    json_safe,
    redact_config,
    redact_headers,
    redact_payload,
    redact_text,
    redact_url,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DebugStore:
    """Write debug records without becoming part of pipeline business logic."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        config: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.requested_root = Path(root)
        self.root = self.requested_root.resolve()
        self.run_id = self._make_run_id()
        self.run_dir = self.root / self.run_id
        self.enabled = False
        self._sequence = 0
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()

        try:
            for relative in (
                "http",
                "extraction",
                "items",
                "ai/calls",
                "tools/calls",
                "summaries",
            ):
                (self.run_dir / relative).mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": "running",
                "created_at": _utc_now(),
                "metadata": redact_payload(metadata or {}),
            }
            self._write_json_unchecked("manifest.json", manifest)
            if config is not None:
                self._write_json_unchecked(
                    "config.redacted.json", redact_config(config)
                )
            self.enabled = True
            self.record_event("run_started", {"run_id": self.run_id})
        except Exception as exc:  # pragma: no cover - depends on filesystem state
            logger.warning("Debug output disabled: %s", exc)

    @staticmethod
    def _make_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4().hex[:8]}"

    def _relative_path(self, *parts: str) -> Path:
        candidate = (self.run_dir.joinpath(*parts)).resolve()
        root = self.run_dir.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Debug path escapes run directory: {candidate}")
        return candidate

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _write_json_unchecked(self, relative: str, payload: Any) -> Path:
        path = self._relative_path(*Path(relative).parts)
        _atomic_write_text(
            path,
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        )
        return path

    def _safe(self, action: str, callback) -> Optional[Path]:
        if not self.enabled:
            return None
        try:
            return callback()
        except Exception as exc:
            logger.warning("Debug write failed during %s: %s", action, exc)
            return None

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def record_event(
        self, event_type: str, payload: Optional[dict[str, Any]] = None
    ) -> None:
        if not self.enabled:
            return

        event = {
            "timestamp": _utc_now(),
            "type": event_type,
            "payload": json_safe(payload or {}),
        }

        def write() -> None:
            path = self._relative_path("events.jsonl")
            with self._event_lock, path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        self._safe("event", write)

    def save_stage(
        self,
        stage: str,
        items: list[Any],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        relative = Path("items") / f"{stage}_items.json"

        def write() -> Path:
            payload = {
                "stage": stage,
                "saved_at": _utc_now(),
                "metadata": json_safe(metadata or {}),
                "items": [json_safe(item) for item in items],
            }
            return self._write_json_unchecked(str(relative), payload)

        result = self._safe(
            f"stage {stage}",
            write,
        )
        self.record_event(
            "stage_saved",
            {"stage": stage, "count": len(items), "path": str(relative)},
        )
        return result

    def save_summary(self, language: str, markdown: str) -> Optional[Path]:
        relative = Path("summaries") / f"summary-{language}.md"

        def write() -> Path:
            path = self._relative_path(*relative.parts)
            _atomic_write_text(path, markdown)
            return path

        return self._safe(
            f"summary {language}",
            write,
        )

    def save_secondary_summary(
        self,
        language: str,
        markdown: str,
    ) -> Optional[Path]:
        relative = Path("summaries") / f"summary-secondary-{language}.md"

        def write() -> Path:
            path = self._relative_path(*relative.parts)
            _atomic_write_text(path, markdown)
            return path

        return self._safe(
            f"secondary summary {language}",
            write,
        )

    def record_http(
        self,
        *,
        request: Any,
        response: Any = None,
        body: bytes | None = None,
        error: Optional[BaseException] = None,
        started_at: Optional[float] = None,
        stage: Optional[str] = None,
    ) -> None:
        sequence = self._next_sequence()
        filename = f"{sequence:06d}"
        context = current_debug_context()
        metadata: dict[str, Any] = {
            "sequence": sequence,
            "recorded_at": _utc_now(),
            "stage": stage or context.stage or "http",
            "item_id": context.item_id,
            "operation": context.operation,
            "language": context.language,
            "block_id": context.block_id,
            "attempt": context.attempt,
            "method": getattr(request, "method", None),
            "url": redact_url(getattr(request, "url", "")),
            "request_headers": redact_headers(getattr(request, "headers", None)),
            "duration_ms": (
                round((time.perf_counter() - started_at) * 1000, 2)
                if started_at is not None
                else None
            ),
        }
        if response is not None:
            metadata.update(
                {
                    "status_code": getattr(response, "status_code", None),
                    "response_headers": redact_headers(
                        getattr(response, "headers", None)
                    ),
                    "final_url": redact_url(getattr(response, "url", "")),
                }
            )
        if error is not None:
            metadata["error"] = {
                "type": type(error).__name__,
                "message": redact_text(error),
            }
        if body is not None:
            body_path = self._relative_path("http", f"{filename}.body")
            metadata["body_file"] = str(body_path.relative_to(self.run_dir))
            metadata["body_bytes"] = len(body)

        def write() -> None:
            if body is not None:
                self._write_bytes(body_path, body)
            self._write_json_unchecked(
                f"http/{filename}.json",
                redact_payload(metadata),
            )

        self._safe("HTTP exchange", write)

    def record_ai_call(
        self,
        *,
        system: str,
        user: str,
        response: Optional[str] = None,
        error: Optional[BaseException] = None,
        duration_ms: Optional[float] = None,
        client: Any = None,
    ) -> None:
        sequence = self._next_sequence()
        context = current_debug_context()
        client_config = getattr(client, "config", None)
        provider = getattr(client_config, "provider", None)
        model = getattr(client, "model", None)
        if provider is None:
            chain = getattr(client, "configs", None) or []
            provider_names = []
            for config in chain:
                candidate = getattr(config, "provider", "")
                provider_names.append(str(getattr(candidate, "value", candidate)))
            provider = ",".join(provider_names) or None
        if model is None:
            chain = getattr(client, "configs", None) or []
            model = (
                ",".join(str(getattr(config, "model", "")) for config in chain)
                or None
            )
        if hasattr(provider, "value"):
            provider = provider.value
        if model is not None:
            model = str(model)
        payload: dict[str, Any] = {
            "sequence": sequence,
            "recorded_at": _utc_now(),
            "stage": context.stage or "ai",
            "item_id": context.item_id,
            "language": context.language,
            "block_id": context.block_id,
            "operation": context.operation,
            "attempt": context.attempt,
            "provider": str(provider) if provider is not None else None,
            "model": model,
            "duration_ms": duration_ms,
            "system": redact_text(system),
            "user": redact_text(user),
            "response": redact_text(response) if response is not None else None,
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": redact_text(error),
            }
        self._safe(
            "AI call",
            lambda: self._write_json_unchecked(
                f"ai/calls/{sequence:06d}.json", redact_payload(payload)
            ),
        )

    def record_tool_call(
        self,
        *,
        request_id: str,
        block_id: str,
        tool: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: Optional[BaseException] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        sequence = self._next_sequence()
        context = current_debug_context()
        payload: dict[str, Any] = {
            "sequence": sequence,
            "recorded_at": _utc_now(),
            "stage": context.stage or "tool",
            "item_id": context.item_id,
            "language": context.language,
            "request_id": request_id,
            "block_id": block_id,
            "tool": tool,
            "arguments": redact_payload(arguments),
            "result": json_safe(result),
            "duration_ms": duration_ms,
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": redact_text(error),
            }
        self._safe(
            "tool call",
            lambda: self._write_json_unchecked(
                f"tools/calls/{sequence:06d}.json", redact_payload(payload)
            ),
        )

    def record_extraction(
        self,
        *,
        url: str,
        extractor: str,
        text: Optional[str],
        error: Optional[BaseException] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        safe_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        status = "failure" if error is not None else (
            "success" if text is not None else "empty"
        )
        metadata: dict[str, Any] = {
            "recorded_at": _utc_now(),
            "url": redact_url(url),
            "extractor": extractor,
            "status": status,
            "duration_ms": duration_ms,
            "text_file": (
                f"extraction/{safe_id}/extracted.txt" if text is not None else None
            ),
        }
        context = current_debug_context()
        metadata.update(
            {
                "stage": context.stage or "extraction",
                "item_id": context.item_id,
                "language": context.language,
                "operation": context.operation,
                "attempt": context.attempt,
            }
        )
        if error is not None:
            metadata["error"] = {
                "type": type(error).__name__,
                "message": redact_text(error),
            }

        def write() -> None:
            directory = self._relative_path("extraction", safe_id)
            directory.mkdir(parents=True, exist_ok=True)
            if text is not None:
                _atomic_write_text(directory / "extracted.txt", text)
            self._write_json_unchecked(
                f"extraction/{safe_id}/metadata.json", redact_payload(metadata)
            )

        self._safe("extraction", write)

    def finish(self, status: str, error: Optional[BaseException] = None) -> None:
        if not self.enabled:
            return
        payload = {
            "status": status,
            "finished_at": _utc_now(),
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": redact_text(error),
            }

        def write() -> None:
            manifest_path = self._relative_path("manifest.json")
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            current.update(payload)
            self._write_json_unchecked("manifest.json", current)

        self._safe("manifest finalization", write)
        self.record_event("run_finished", payload)
