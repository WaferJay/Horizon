"""Decorator-style observers for existing Horizon components."""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from ..ai.client import AIClient
from ..extractors.base import BaseExtractor
from .context import current_debug_context, debug_scope
from .store import DebugStore


class RecordingAsyncClient(httpx.AsyncClient):
    """Record transport failures that do not produce an HTTP response."""

    def __init__(self, store: DebugStore, *args: Any, **kwargs: Any):
        self.store = store
        super().__init__(*args, **kwargs)

    async def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        started = request.extensions.get("horizon_debug_started")
        try:
            return await super().send(request, *args, **kwargs)
        except Exception as exc:
            self.store.record_http(
                request=request,
                error=exc,
                started_at=started,
                stage=request.extensions.get(
                    "horizon_debug_stage", current_debug_context().stage or "http"
                ),
            )
            raise


class RecordingAIClient(AIClient):
    """Record AI calls while preserving the wrapped client's interface."""

    def __init__(self, wrapped: AIClient, store: DebugStore):
        self.wrapped = wrapped
        self.store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        started = time.perf_counter()
        try:
            response = await self.wrapped.complete(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self.store.record_ai_call(
                system=system,
                user=user,
                error=exc,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                client=self.wrapped,
            )
            raise

        self.store.record_ai_call(
            system=system,
            user=user,
            response=response,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            client=self.wrapped,
        )
        return response


class RecordingToolRegistry:
    """Observe tool calls without changing the built-in registry."""

    def __init__(self, wrapped: Any, store: DebugStore):
        self.wrapped = wrapped
        self.store = store

    @property
    def names(self) -> set[str]:
        return self.wrapped.names

    async def execute(
        self,
        request_id: str,
        block_id: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> Any:
        started = time.perf_counter()
        try:
            result = await self.wrapped.execute(
                request_id=request_id,
                block_id=block_id,
                tool=tool,
                arguments=arguments,
            )
        except Exception as exc:
            self.store.record_tool_call(
                request_id=request_id,
                block_id=block_id,
                tool=tool,
                arguments=arguments,
                error=exc,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise

        self.store.record_tool_call(
            request_id=request_id,
            block_id=block_id,
            tool=tool,
            arguments=arguments,
            result={
                "request_id": result.request_id,
                "block_id": result.block_id,
                "tool": result.tool,
                "results": result.results,
            },
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return result


class RecordingExtractor:
    """Observe an extractor invocation and preserve its return value."""

    def __init__(self, name: str, wrapped: BaseExtractor, store: DebugStore):
        self.name = name
        self.wrapped = wrapped
        self.store = store

    async def extract(self, url: str, client: Any) -> Optional[str]:
        started = time.perf_counter()
        with debug_scope(stage="extraction", operation="extract"):
            try:
                text = await self.wrapped.extract(url, client)
            except Exception as exc:
                self.store.record_extraction(
                    url=url,
                    extractor=self.name,
                    text=None,
                    error=exc,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                raise
            self.store.record_extraction(
                url=url,
                extractor=self.name,
                text=text,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return text


class RecordingExtractorRegistry:
    """Registry adapter that decorates extractors on lookup."""

    def __init__(self, wrapped: Any, store: DebugStore):
        self.wrapped = wrapped
        self.store = store
        self._cache: dict[str, RecordingExtractor] = {}

    def get(self, name: str) -> Optional[RecordingExtractor]:
        if name in self._cache:
            return self._cache[name]
        extractor = self.wrapped.get(name)
        if extractor is None:
            return None
        observed = RecordingExtractor(name, extractor, self.store)
        self._cache[name] = observed
        return observed


def make_http_event_hooks(
    store: DebugStore, stage: str = "fetch"
) -> dict[str, list[Any]]:
    """Build httpx hooks that observe responses without changing callers."""

    async def on_request(request: Any) -> None:
        request.extensions["horizon_debug_started"] = time.perf_counter()
        request.extensions["horizon_debug_stage"] = (
            current_debug_context().stage or stage
        )

    async def on_response(response: Any) -> None:
        request = response.request
        started = request.extensions.get("horizon_debug_started")
        try:
            body = await response.aread()
            store.record_http(
                request=request,
                response=response,
                body=body,
                started_at=started,
                stage=request.extensions.get("horizon_debug_stage", stage),
            )
        except Exception as exc:
            store.record_http(
                request=request,
                response=response,
                error=exc,
                started_at=started,
                stage=request.extensions.get("horizon_debug_stage", stage),
            )

    return {"request": [on_request], "response": [on_response]}
