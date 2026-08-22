"""Context metadata shared by debug decorators."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator, Optional


@dataclass(frozen=True)
class DebugContext:
    """Metadata attached to an operation being observed."""

    stage: Optional[str] = None
    item_id: Optional[str] = None
    language: Optional[str] = None
    block_id: Optional[str] = None
    operation: Optional[str] = None
    attempt: Optional[int] = None


_CURRENT_CONTEXT: ContextVar[DebugContext] = ContextVar(
    "horizon_debug_context", default=DebugContext()
)


def current_debug_context() -> DebugContext:
    """Return the metadata for the current async task."""

    return _CURRENT_CONTEXT.get()


@contextmanager
def debug_scope(
    *,
    stage: Optional[str] = None,
    item_id: Optional[str] = None,
    language: Optional[str] = None,
    block_id: Optional[str] = None,
    operation: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Iterator[DebugContext]:
    """Temporarily add metadata to the current debug context.

    ``None`` means "leave the inherited value unchanged".  Nested scopes are
    task-local through ``contextvars`` and therefore safe for concurrent
    analysis/enrichment tasks.
    """

    previous = _CURRENT_CONTEXT.get()
    current = replace(
        previous,
        stage=stage if stage is not None else previous.stage,
        item_id=item_id if item_id is not None else previous.item_id,
        language=language if language is not None else previous.language,
        block_id=block_id if block_id is not None else previous.block_id,
        operation=operation if operation is not None else previous.operation,
        attempt=attempt if attempt is not None else previous.attempt,
    )
    token = _CURRENT_CONTEXT.set(current)
    try:
        yield current
    finally:
        _CURRENT_CONTEXT.reset(token)
