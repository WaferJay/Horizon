from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.models import (
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
        async def complete(self, *, system, user):
            return '{"items":[{"id":"low","title":"Short title","summary":"Short summary"}]}'

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
