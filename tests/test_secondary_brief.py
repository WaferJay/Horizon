from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.ai.secondary_brief import SecondaryBriefGenerator
from src.debug import DebugStore, RecordingAIClient
from src.models import (
    AIStage,
    ClassificationResult,
    ContentAnalysis,
    ContentItem,
    ProcessingResult,
    SourceType,
)
from src.services.secondary_brief import SecondaryBriefSelector, SecondaryBriefService


def _item(item_id: str, score: float, url: str | None = None) -> ContentItem:
    return ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=f"Item {item_id}",
        url=url or f"https://example.com/{item_id}",
        content="content",
        author="tester",
        published_at=datetime.now(timezone.utc),
        profile="tech-news",
        processing=ProcessingResult(
            classification=ClassificationResult(
                profile="tech-news", method="source_override"
            ),
            analysis=ContentAnalysis(score=score, reason="test", summary=item_id),
        ),
    )


def _selector(deduplicator):
    config = SimpleNamespace(
        digest=SimpleNamespace(
            secondary_brief=SimpleNamespace(min_score=2.0),
        ),
        processing=SimpleNamespace(
            profile_settings={
                "tech-news": SimpleNamespace(threshold=7.0, topic_dedup=True),
            }
        ),
    )
    return SecondaryBriefSelector(
        config=config,
        default_profile="tech-news",
        topic_deduplicator=deduplicator,
        url_key=lambda value: value.rstrip("/"),
    )


def test_selector_keeps_scored_items_between_floor_and_main_threshold():
    async def no_dedup(items, *, log=True):
        return items

    selector = _selector(no_dedup)
    main = _item("main", 9.0)
    same_url = _item("same-url", 5.0, url=str(main.url) + "/")
    candidates = [main, _item("low", 6.0), _item("too-low", 1.0), same_url]

    result = asyncio.run(selector.select(candidates, [main], log=False))

    assert [item.id for item in result.items] == ["low"]


def test_selector_reuses_topic_dedup_and_records_dropped_secondary_ids():
    async def keep_primary(items, *, log=True):
        return items[::2]

    selector = _selector(keep_primary)
    main = _item("main", 9.0)
    low_a = _item("low-a", 6.0)
    low_b = _item("low-b", 5.0)

    result = asyncio.run(selector.select([main, low_a, low_b], [main], log=False))

    assert [item.id for item in result.items] == ["low-b"]
    assert result.duplicate_ids == ["low-a"]


def test_service_always_saves_standalone_and_optionally_appends(tmp_path: Path):
    class FakeClient:
        async def complete(self, *, system, user, **kwargs):
            return '{"title":"Short title","summary":"Short summary"}'

    class Storage:
        def save_secondary_summary(self, date, markdown, *, language):
            path = tmp_path / f"horizon-secondary-{date}-{language}.md"
            path.write_text(markdown, encoding="utf-8")
            return path

    config = SimpleNamespace(
        digest=SimpleNamespace(
            secondary_brief=SimpleNamespace(min_score=2.0, append_to_main=True)
        )
    )
    service = SecondaryBriefService(
        config=config,
        storage=Storage(),
        client=FakeClient(),
    )

    output = asyncio.run(
        service.build(
            [_item("low", 4.0)],
            "# Main report",
            "2026-08-23",
            language="zh",
        )
    )

    assert output.path is not None
    assert output.path.name == "horizon-secondary-2026-08-23-zh.md"
    assert output.path.exists()
    assert "# Main report" in output.report
    assert "## 补充资讯" in output.report
    assert "Short summary" in output.summary


def test_generator_sends_one_id_free_prompt_per_item_and_associates_results():
    first = _item("rss:" + "a" * 120, 6.0)
    second = _item("rss:" + "b" * 120, 5.0)
    first.title = "Alpha headline"
    second.title = "Beta headline"
    first.processing.analysis.summary = "Alpha source summary"
    second.processing.analysis.summary = "Beta source summary"

    class FakeClient:
        config = SimpleNamespace(secondary_concurrency=2)

        def __init__(self):
            self.calls = []

        async def complete(self, *, system, user, **kwargs):
            self.calls.append((system, user))
            if "Alpha headline" in user:
                await asyncio.sleep(0.01)
                return '{"title":"Alpha short","summary":"Alpha localized"}'
            return '{"title":"Beta short","summary":"Beta localized"}'

    client = FakeClient()
    generated = asyncio.run(
        SecondaryBriefGenerator(client).generate([first, second], language="zh")
    )

    assert len(client.calls) == 2
    alpha_call = next(call for call in client.calls if "Alpha headline" in call[1])
    beta_call = next(call for call in client.calls if "Beta headline" in call[1])
    assert "Beta headline" not in alpha_call[1]
    assert "Beta source summary" not in alpha_call[1]
    assert "Alpha headline" not in beta_call[1]
    assert "Alpha source summary" not in beta_call[1]
    for system, user in client.calls:
        assert first.id not in system + user
        assert second.id not in system + user
        assert '"items"' not in system
    assert generated[first.id].title == "Alpha short"
    assert generated[first.id].summary == "Alpha localized"
    assert generated[second.id].title == "Beta short"
    assert generated[second.id].summary == "Beta localized"


def test_generator_retries_transport_errors_three_times_then_isolates_failure(
    monkeypatch,
):
    from tenacity import wait_none

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        SecondaryBriefGenerator._complete.retry,
        "wait",
        wait_none(),
    )
    client = FakeClient()
    generated = asyncio.run(
        SecondaryBriefGenerator(client).generate([_item("failed", 4.0)], language="en")
    )

    assert generated == {}
    assert client.calls == 3


def test_generator_repairs_invalid_json_and_stops_after_three_attempts():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            return "not valid json"

    client = FakeClient()
    generated = asyncio.run(
        SecondaryBriefGenerator(client).generate([_item("invalid", 4.0)], language="en")
    )

    assert generated == {}
    assert len(client.calls) == 3
    assert all(
        call["stage"] == AIStage.SECONDARY_BRIEF for call in client.calls
    )
    assert "previous response did not satisfy" not in client.calls[0]["user"]
    for call in client.calls[1:]:
        assert call["temperature"] == 0
        assert "not valid json" in call["user"]
        assert "previous response did not satisfy" in call["user"]


def test_permanent_item_failure_falls_back_without_interrupting_report_order():
    bad = _item("bad", 6.0)
    good = _item("good", 5.0)
    bad.title = "Bad original title"
    good.title = "Good original title"
    bad.processing.analysis.summary = "Bad analysis fallback"
    good.processing.analysis.summary = "Good source summary"

    class FakeClient:
        config = SimpleNamespace(secondary_concurrency=2)

        async def complete(self, **kwargs):
            if "Bad original title" in kwargs["user"]:
                return "invalid"
            return '{"title":"Good localized title","summary":"Good localized summary"}'

    config = SimpleNamespace(
        digest=SimpleNamespace(
            secondary_brief=SimpleNamespace(min_score=2.0, append_to_main=False)
        )
    )
    service = SecondaryBriefService(config=config, storage=None, client=FakeClient())
    output = asyncio.run(
        service.build([bad, good], "# Main", "2026-08-31", language="en")
    )

    assert output.report == "# Main"
    assert "Bad original title" in output.summary
    assert "Bad analysis fallback" in output.summary
    assert "Good localized title" in output.summary
    assert "Good localized summary" in output.summary
    assert output.summary.index("Bad original title") < output.summary.index(
        "Good localized title"
    )


def test_generator_limits_concurrent_requests():
    class FakeClient:
        config = SimpleNamespace(secondary_concurrency=2)

        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def complete(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return '{"title":"Short","summary":"Summary"}'
            finally:
                self.active -= 1

    client = FakeClient()
    items = [_item(f"item-{index}", 4.0) for index in range(6)]
    generated = asyncio.run(
        SecondaryBriefGenerator(client).generate(items, language="en")
    )

    assert len(generated) == len(items)
    assert client.max_active == 2


def test_generator_empty_input_does_not_call_model():
    class FakeClient:
        async def complete(self, **kwargs):
            raise AssertionError("model must not be called")

    generated = asyncio.run(SecondaryBriefGenerator(FakeClient()).generate([], language="en"))

    assert generated == {}


def test_secondary_debug_calls_include_item_operation_and_structured_attempt(
    tmp_path: Path,
):
    class FakeClient:
        config = SimpleNamespace(provider="test", secondary_concurrency=2)
        model = "test-model"

        def __init__(self):
            self.responses = iter(
                [
                    "invalid",
                    '{"title":"","summary":"still invalid"}',
                    '{"title":"Localized","summary":"Localized summary"}',
                ]
            )

        async def complete(self, **kwargs):
            return next(self.responses)

    store = DebugStore(tmp_path)
    client = RecordingAIClient(FakeClient(), store)
    generated = asyncio.run(
        SecondaryBriefGenerator(client).generate(
            [_item("debug-item", 4.0)],
            language="zh",
        )
    )

    assert generated["debug-item"].title == "Localized"
    calls = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((store.run_dir / "ai/calls").glob("*.json"))
    ]
    assert [call["attempt"] for call in calls] == [1, 2, 3]
    assert {call["item_id"] for call in calls} == {"debug-item"}
    assert {call["operation"] for call in calls} == {"summarize"}
    assert {call["stage"] for call in calls} == {"secondary_brief"}
