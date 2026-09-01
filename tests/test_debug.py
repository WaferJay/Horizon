import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx

from src._cli import add_debug_dir_argument
from src.debug import (
    DebugStore,
    RecordingAIClient,
    RecordingAsyncClient,
    RecordingExtractor,
    RecordingToolRegistry,
    debug_scope,
    make_http_event_hooks,
)
from src.extractors.base import BaseExtractor
from src.models import AIStage, ContentItem, SourceType
from src.processing.tools import ToolResult


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_debug_store_persists_stages_and_redacts_credentials(tmp_path):
    store = DebugStore(
        tmp_path,
        config={"ai": {"api_key": "secret", "api_key_env": "API_KEY"}},
        metadata={"authorization": "Bearer secret"},
    )
    item = ContentItem(
        id="rss:test:1",
        source_type=SourceType.RSS,
        title="A title",
        url="https://example.test/news?token=secret",
        content="Body",
        published_at=datetime.now(timezone.utc),
    )

    store.save_stage("raw", [item], metadata={"token": "secret"})
    with debug_scope(stage="analysis", item_id=item.id, operation="analyze"):
        store.record_ai_call(
            system="authorization: secret",
            user="https://example.test/?api_key=secret",
            response="{}",
        )
    store.finish("success")

    assert store.enabled
    assert _read_json(store.run_dir / "manifest.json")["status"] == "success"
    assert (
        _read_json(store.run_dir / "config.redacted.json")["ai"]["api_key"]
        == "[REDACTED]"
    )
    stage = _read_json(store.run_dir / "items/raw_items.json")
    assert stage["items"][0]["url"] == "https://example.test/news?token=%5BREDACTED%5D"
    ai_call = _read_json(next((store.run_dir / "ai/calls").glob("*.json")))
    assert "secret" not in json.dumps(ai_call)


def test_http_hooks_save_response_body_and_redact_url(tmp_path):
    store = DebugStore(tmp_path)

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/xml", "Authorization": "secret"},
            content=b"<rss>body</rss>",
            request=request,
        )

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            event_hooks=make_http_event_hooks(store),
        ) as client:
            response = await client.get("https://example.test/feed?api_key=secret")
            assert response.text == "<rss>body</rss>"

    asyncio.run(fetch())

    record_path = next((store.run_dir / "http").glob("*.json"))
    record = _read_json(record_path)
    assert record["url"].endswith("api_key=%5BREDACTED%5D")
    assert (store.run_dir / record["body_file"]).read_bytes() == b"<rss>body</rss>"
    assert "Authorization" not in record["response_headers"]


def test_recording_http_client_captures_transport_failures(tmp_path):
    store = DebugStore(tmp_path)

    def handler(request):
        raise httpx.ConnectError("connection failed", request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with RecordingAsyncClient(
            store,
            transport=transport,
            event_hooks=make_http_event_hooks(store),
        ) as client:
            try:
                await client.get("https://example.test/failing")
            except httpx.ConnectError:
                return
        raise AssertionError("request should fail")

    asyncio.run(fetch())

    record = _read_json(next((store.run_dir / "http").glob("*.json")))
    assert "status_code" not in record
    assert record["error"]["type"] == "ConnectError"


def test_recording_adapters_preserve_results_and_context(tmp_path):
    store = DebugStore(tmp_path)

    class FakeAI:
        config = SimpleNamespace(provider="llama.cpp")
        model = "test-model"

        def __init__(self):
            self.calls = []

        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            return "response"

    class FakeTools:
        names = {"web_search"}

        async def execute(self, **kwargs):
            return ToolResult(
                request_id=kwargs["request_id"],
                block_id=kwargs["block_id"],
                tool=kwargs["tool"],
                results=[{"title": "Result", "url": "https://example.test"}],
            )

    class FakeExtractor(BaseExtractor):
        async def extract(self, url, client):
            return "article text"

    async def exercise():
        fake_ai = FakeAI()
        with debug_scope(stage="analysis", item_id="item-1", attempt=1):
            response = await RecordingAIClient(fake_ai, store).complete(
                system="system",
                user="user",
                stage=AIStage.ANALYSIS,
            )
        assert response == "response"
        assert fake_ai.calls[0]["stage"] == AIStage.ANALYSIS

        with debug_scope(stage="enrichment", item_id="item-1", language="zh"):
            result = await RecordingToolRegistry(FakeTools(), store).execute(
                request_id="tool-1",
                block_id="background",
                tool="web_search",
                arguments={"query": "test"},
            )
        assert result.results[0]["title"] == "Result"

        extracted = await RecordingExtractor("test", FakeExtractor(), store).extract(
            "https://example.test/article", None
        )
        assert extracted == "article text"

    asyncio.run(exercise())

    assert list((store.run_dir / "ai/calls").glob("*.json"))
    tool_call = _read_json(next((store.run_dir / "tools/calls").glob("*.json")))
    assert tool_call["item_id"] == "item-1"
    extraction_dir = next((store.run_dir / "extraction").iterdir())
    assert (
        (extraction_dir / "extracted.txt").read_text(encoding="utf-8")
        == "article text"
    )


def test_debug_write_failure_is_best_effort(tmp_path, monkeypatch):
    store = DebugStore(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_json_unchecked", fail)
    assert store.save_stage("raw", []) is None


def test_debug_cli_argument_is_opt_in():
    import argparse

    parser = argparse.ArgumentParser()
    add_debug_dir_argument(parser)
    assert parser.parse_args([]).debug_dir is None
    assert parser.parse_args(["--debug-dir", "/tmp/debug"]).debug_dir == "/tmp/debug"
