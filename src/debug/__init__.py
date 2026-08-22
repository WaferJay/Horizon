"""Opt-in debug observers and persistent run artifacts."""

from .context import DebugContext, current_debug_context, debug_scope
from .store import DebugStore
from .wrappers import (
    RecordingAIClient,
    RecordingAsyncClient,
    RecordingExtractor,
    RecordingExtractorRegistry,
    RecordingToolRegistry,
    make_http_event_hooks,
)

__all__ = [
    "DebugContext",
    "DebugStore",
    "RecordingAIClient",
    "RecordingAsyncClient",
    "RecordingExtractor",
    "RecordingExtractorRegistry",
    "RecordingToolRegistry",
    "current_debug_context",
    "debug_scope",
    "make_http_event_hooks",
]
