"""Redaction helpers for persisted debug data."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "set_cookie",
    "signature",
    "token",
}

_SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}

_SECRET_TEXT_RE = re.compile(
    r"(?i)(\b(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password|secret|token)\b\s*[:=]\s*[\"']?)([^\s,\"'}]+)"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s]+")


def _normalized_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_")


def is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized.endswith("_env"):
        return False
    return normalized in _SENSITIVE_KEYS or any(
        part in normalized
        for part in (
            "api_key",
            "access_token",
            "refresh_token",
            "secret",
            "password",
        )
    ) or normalized.endswith(("_token", "_key"))


def redact_url(value: object) -> str:
    """Redact sensitive query parameters while preserving the URL shape."""

    raw = str(value)
    try:
        parsed = urlsplit(raw)
        query = [
            (key, REDACTED if is_sensitive_key(key) else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    except ValueError:
        return raw


def redact_headers(headers: Mapping[str, object] | None) -> dict[str, str]:
    """Return headers without credential-bearing values."""

    if not headers:
        return {}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if (
            str(key).lower() not in _SENSITIVE_HEADERS
            and not is_sensitive_key(key)
        )
    }


def redact_text(value: object) -> str:
    """Redact common inline credential forms from prompts and metadata."""

    text = str(value)
    text = _BEARER_RE.sub(r"\1" + REDACTED, text)
    return _SECRET_TEXT_RE.sub(r"\1" + REDACTED, text)


def redact_payload(value: Any) -> Any:
    """Recursively redact credential-like mapping fields."""

    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            return redact_url(value)
        return redact_text(value)
    return value


def redact_config(value: Any) -> Any:
    """Redact a config model/dict before persisting it."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return redact_payload(value)


def json_safe(value: Any) -> Any:
    """Convert a value to a JSON-safe, redacted representation."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            return redact_url(value)
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return redact_text(value)
