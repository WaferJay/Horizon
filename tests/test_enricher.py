import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai.enricher import ContentEnricher, ToolPlan
from src.models import (
    AIStage,
    ClassificationResult,
    ContentAnalysis,
    ContentArtifact,
    ContentBlock,
    ContentItem,
    ProcessingResult,
    SourceType,
)
from src.processing import ProfileRegistry
from src.processing.tools import ToolInputError, ToolResult, WebSearchTool


PROFILES = ProfileRegistry.load(
    Path(__file__).resolve().parents[1] / "profiles", "tech-news"
)


def _profiles_with_international_background_search() -> ProfileRegistry:
    """Build the mixed-permission profile required by tool-validation tests."""
    international = PROFILES.get("international-news")
    blocks = [
        block.model_copy(update={"tools": ["web_search"]})
        if block.id == "background"
        else block
        for block in international.definition.enrichment.blocks
    ]
    enrichment = international.definition.enrichment.model_copy(
        update={"blocks": blocks}
    )
    definition = international.definition.model_copy(
        update={"enrichment": enrichment}
    )
    profiles = {profile.id: profile for profile in PROFILES.profiles}
    profiles[international.id] = replace(international, definition=definition)
    return ProfileRegistry(profiles, PROFILES.default_profile)


INTERNATIONAL_TOOL_PROFILES = _profiles_with_international_background_search()


def make_item() -> ContentItem:
    return ContentItem(
        id="rss:test:item",
        source_type=SourceType.RSS,
        title="A technical release",
        url="https://example.com/item",
        content="A project released a new architecture.",
        published_at=datetime.now(timezone.utc),
        profile="tech-news",
        processing=ProcessingResult(
            classification=ClassificationResult(
                profile="tech-news", method="source_override"
            ),
            analysis=ContentAnalysis(
                score=8.5,
                reason="Important release",
                summary="A new architecture was released.",
                tags=["systems"],
            ),
        ),
    )


class FakeTools:
    names = {"web_search"}

    async def execute(self, request_id, block_id, tool, arguments):
        assert block_id == "background"
        assert tool == "web_search"
        assert arguments == {"query": "project architecture"}
        return ToolResult(
            request_id=request_id,
            block_id=block_id,
            tool=tool,
            results=[
                {
                    "title": "Project documentation",
                    "url": "https://docs.example.com/project",
                    "text": "Architecture background.",
                }
            ],
        )


class RecordingTools:
    names = {"web_search"}

    def __init__(self, *, results=None, error=None):
        self.calls = []
        self.results = [] if results is None else results
        self.error = error

    async def execute(self, request_id, block_id, tool, arguments):
        self.calls.append(
            {
                "request_id": request_id,
                "block_id": block_id,
                "tool": tool,
                "arguments": arguments,
            }
        )
        if self.error:
            raise self.error
        return ToolResult(
            request_id=request_id,
            block_id=block_id,
            tool=tool,
            results=self.results,
        )


def artifact_response(block_ids, title="Article update"):
    return json.dumps(
        {
            "title": title,
            "blocks": [
                {
                    "id": block_id,
                    "title": block_id.replace("_", " ").title(),
                    "content": f"Supported content for {block_id}.",
                    "source_refs": [],
                }
                for block_id in block_ids
            ],
        }
    )


def test_enrichment_generates_blocks_and_validated_sources():
    responses = iter(
        [
            json.dumps(
                {
                    "tool_requests": [
                        {
                            "block_id": "background",
                            "tool": "web_search",
                            "arguments": {"query": "project architecture"},
                            "purpose": "Explain the existing architecture",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "title": "新架構發佈",
                    "blocks": [
                        {
                            "id": "summary",
                            "title": "摘要",
                            "content": "項目發佈了新的架構，它改變了系統設計，並採用了新的邊界。",
                            "source_refs": [],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "title": "新架構發佈",
                    "blocks": [
                        {
                            "id": "summary",
                            "type": "section",
                            "title": "摘要",
                            "content": "项目发布了新的架构，它改变了系统设计，并采用了新的边界。",
                            "source_refs": [],
                        },
                        {
                            "id": "background",
                            "type": "section",
                            "title": "未隔离的背景",
                            "content": "这个版本应被丢弃。",
                            "source_refs": [],
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "title": "",
                    "block": {
                        "id": "background",
                        "type": "section",
                        "title": "背景",
                        "content": "旧架构的背景信息。",
                        "source_refs": ["tool-1"],
                    },
                }
            ),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    item = make_item()
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["zh"],
        tools=FakeTools(),
    )
    asyncio.run(enricher._enrich_item(item))

    artifact = item.processing.artifacts["zh"]
    assert artifact.title == "新架构发布"
    assert artifact.blocks[0].content == "项目发布了新的架构，它改变了系统设计，并采用了新的边界。"
    assert [block.id for block in artifact.blocks] == [
        "summary",
        "background",
    ]
    assert artifact.blocks[-1].title == "背景"
    assert artifact.blocks[0].primary is True
    assert artifact.blocks[1].primary is False
    assert artifact.blocks[-1].source_refs == ["tool-1-1"]
    assert artifact.sources[0].url == "https://docs.example.com/project"
    assert len(requests) == 4
    assert "explicitly mentioned in the item" in requests[0]["system"]
    assert "Treat the source item as the primary account" in requests[1]["system"]
    assert "Simplified Chinese (language tag `zh`)" in requests[1]["system"]
    assert "Treat the source item as the primary account" in requests[3]["system"]
    assert "https://docs.example.com/project" not in requests[1]["user"]
    assert "https://docs.example.com/project" in requests[3]["user"]


def test_enrichment_repairs_disallowed_tool_request_and_continues(caplog):
    required_blocks = (
        "summary",
        "background",
        "key_actors",
        "domestic_context",
        "international_impact",
        "uncertainty",
    )
    responses = iter(
        [
            json.dumps(
                {
                    "tool_requests": [
                        {
                            "block_id": "key_actors",
                            "tool": "web_search",
                            "arguments": {"query": "unapproved"},
                            "purpose": "Research the actors",
                        }
                    ]
                }
            ),
            json.dumps({"tool_requests": []}),
            artifact_response(required_blocks, "International update"),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    tools = RecordingTools()
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        INTERNATIONAL_TOOL_PROFILES,
        ["en"],
        tools=tools,
    )
    item = make_item()
    item.profile = "international-news"
    item.processing.classification.profile = "international-news"

    with caplog.at_level(logging.WARNING):
        asyncio.run(enricher._enrich_item(item))

    assert tools.calls == []
    assert len(requests) == 3
    assert "not allowed for block key_actors" in requests[1]["user"]
    assert "No tool results are available" in requests[2]["user"]
    assert "Do not retry tool use" in requests[2]["user"]
    assert [
        block.id for block in item.processing.artifacts["en"].blocks
    ] == list(required_blocks)
    assert "requesting one correction" in caplog.text


def test_enrichment_keeps_valid_requests_from_mixed_corrected_plan(caplog):
    responses = iter(
        [
            json.dumps(
                {
                    "tool_requests": [
                        {
                            "block_id": "key_actors",
                            "tool": "web_search",
                            "arguments": {"query": "actors"},
                            "purpose": "Research the actors",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "tool_requests": [
                        {
                            "block_id": "background",
                            "tool": "web_search",
                            "arguments": {"query": "project architecture"},
                            "purpose": "Research background",
                        },
                        {
                            "block_id": "key_actors",
                            "tool": "web_search",
                            "arguments": {"query": "actors"},
                            "purpose": "Research the actors",
                        },
                        {
                            "block_id": "unknown_block",
                            "tool": "web_search",
                            "arguments": {"query": "unknown"},
                            "purpose": "Research an unknown block",
                        },
                    ]
                }
            ),
        ]
    )

    async def complete(**kwargs):
        return next(responses)

    tools = RecordingTools()
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        INTERNATIONAL_TOOL_PROFILES,
        ["en"],
        tools=tools,
    )

    with caplog.at_level(logging.WARNING):
        results = asyncio.run(
            enricher._plan_and_execute_tools(
                make_item(),
                INTERNATIONAL_TOOL_PROFILES.get("international-news"),
            )
        )

    assert [result.block_id for result in results] == ["background"]
    assert [call["block_id"] for call in tools.calls] == ["background"]
    assert "Ignoring invalid enrichment tool request" in caplog.text
    assert "unknown block unknown_block" in caplog.text


def test_enrichment_falls_back_when_tool_plan_remains_malformed(caplog):
    responses = iter(
        [
            "[]",
            "[]",
            "[]",
            artifact_response(("summary", "background")),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    with caplog.at_level(logging.WARNING):
        item = make_item()
        asyncio.run(enricher._enrich_item(item))

    assert len(requests) == 4
    assert item.processing.artifacts["en"].title == "Article update"
    assert "continuing without tools" in caplog.text


def test_enrichment_repairs_malformed_tool_plan_once():
    responses = iter(
        [
            "[]",
            json.dumps({"tool_requests": []}),
            json.dumps(
                {
                    "title": "Technical release",
                    "blocks": [
                        {
                            "id": "summary",
                            "type": "section",
                            "title": "Summary",
                            "content": "A complete summary.",
                            "source_refs": [],
                        },
                        {
                            "id": "background",
                            "type": "section",
                            "title": "Background",
                            "content": "Context for the release.",
                            "source_refs": [],
                        }
                    ],
                }
            ),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    item = make_item()
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    asyncio.run(enricher._enrich_item(item))

    assert len(requests) == 3
    assert all(request["temperature"] == 0 for request in requests)
    assert all(request["stage"] == AIStage.ENRICHMENT for request in requests)
    assert requests[1]["temperature"] == 0
    assert item.processing.artifacts["en"].blocks[0].id == "summary"


def test_enrichment_ignores_invalid_tool_arguments(caplog):
    async def complete(**kwargs):
        return json.dumps(
            {
                "tool_requests": [
                    {
                        "block_id": "background",
                        "tool": "web_search",
                        "arguments": {"query": ""},
                        "purpose": "Research background",
                    }
                ]
            }
        )

    tools = RecordingTools(
        error=ToolInputError("web_search requires a non-empty query")
    )
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=tools,
    )

    with caplog.at_level(logging.WARNING):
        results = asyncio.run(
            enricher._plan_and_execute_tools(
                make_item(),
                PROFILES.get("tech-news"),
            )
        )

    assert results == []
    assert len(tools.calls) == 1
    assert "Ignoring invalid arguments" in caplog.text


def test_web_search_uses_recoverable_error_for_invalid_arguments():
    with pytest.raises(ToolInputError, match="non-empty query"):
        asyncio.run(WebSearchTool().execute({"query": " "}))


def test_enrichment_does_not_hide_unexpected_tool_errors():
    async def complete(**kwargs):
        return json.dumps(
            {
                "tool_requests": [
                    {
                        "block_id": "background",
                        "tool": "web_search",
                        "arguments": {"query": "project architecture"},
                        "purpose": "Research background",
                    }
                ]
            }
        )

    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=RecordingTools(error=RuntimeError("tool implementation failed")),
    )

    with pytest.raises(RuntimeError, match="tool implementation failed"):
        asyncio.run(
            enricher._plan_and_execute_tools(
                make_item(),
                PROFILES.get("tech-news"),
            )
        )


def test_enrichment_explains_empty_tool_results_to_block_model():
    responses = iter(
        [
            json.dumps(
                {
                    "tool_requests": [
                        {
                            "block_id": "background",
                            "tool": "web_search",
                            "arguments": {"query": "project architecture"},
                            "purpose": "Research background",
                        }
                    ]
                }
            ),
            artifact_response(("summary",)),
            json.dumps(
                {
                    "title": "",
                    "block": {
                        "id": "background",
                        "title": "Background",
                        "content": "The source provides limited background.",
                        "source_refs": [],
                    },
                }
            ),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    item = make_item()
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=RecordingTools(results=[]),
    )

    asyncio.run(enricher._enrich_item(item))

    assert "No results were returned by this tool request" in requests[2]["user"]
    assert "do not invent external facts" in requests[2]["user"]
    assert item.processing.artifacts["en"].sources == []


def test_enrichment_json_repair_includes_error_and_previous_response():
    responses = iter(
        [
            "not JSON at all",
            "still not JSON",
            json.dumps({"tool_requests": []}),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    result = asyncio.run(
        enricher._complete_model(
            ToolPlan,
            system="Return a tool plan as JSON.",
            user="Analyze this item.",
            error_message="Invalid tool plan",
        )
    )

    assert result.tool_requests == []
    assert len(requests) == 3
    assert "could not parse a valid JSON response" in requests[1]["user"]
    assert "not JSON at all" in requests[1]["user"]
    assert "still not JSON" in requests[2]["user"]
    assert "not JSON at all" not in requests[2]["user"]


def test_enrichment_json_repair_stops_after_three_attempts():
    responses = iter(["invalid 1", "invalid 2", "invalid 3", "invalid 4"])
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    with pytest.raises(ValueError, match="Invalid tool plan"):
        asyncio.run(
            enricher._complete_model(
                ToolPlan,
                system="Return a tool plan as JSON.",
                user="Analyze this item.",
                error_message="Invalid tool plan",
            )
        )

    assert len(requests) == 3


def test_enrichment_repairs_empty_blog_block_once():
    responses = iter(
        [
            json.dumps(
                {
                    "title": "A technical story",
                    "blocks": [
                        {
                            "id": "background",
                            "title": " ",
                            "content": "",
                            "source_refs": [],
                        },
                        {
                            "id": "solution",
                            "title": "Solution and results",
                            "content": "The author explains the implementation.",
                            "source_refs": [],
                        },
                        {
                            "id": "takeaway",
                            "title": "Takeaway",
                            "content": "The approach is useful in bounded cases.",
                            "source_refs": [],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "title": "A technical story",
                    "blocks": [
                        {
                            "id": "background",
                            "title": "Background",
                            "content": "The author frames the original problem and constraints.",
                            "source_refs": [],
                        },
                        {
                            "id": "solution",
                            "title": "Solution and results",
                            "content": "The author explains the implementation and its effects.",
                            "source_refs": [],
                        },
                        {
                            "id": "takeaway",
                            "title": "Takeaway",
                            "content": "The approach is useful in bounded cases.",
                            "source_refs": [],
                        }
                    ],
                }
            ),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    item = make_item()
    item.profile = "tech-blog"
    item.processing.classification.profile = "tech-blog"
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    asyncio.run(enricher._enrich_item(item))

    assert len(requests) == 2
    assert requests[1]["temperature"] == 0
    assert "corrected JSON object" in requests[1]["user"]
    assert [
        block.id for block in item.processing.artifacts["en"].blocks
    ] == ["background", "solution", "takeaway"]
    assert all(
        not block.primary for block in item.processing.artifacts["en"].blocks
    )


def test_enrichment_repairs_schema_type_used_as_blog_block_id():
    responses = iter(
        [
            json.dumps(
                {
                    "title": "A technical story",
                    "blocks": [
                        {
                            "id": "section",
                            "type": "section",
                            "title": "Background",
                            "content": "A complete but misidentified background block.",
                            "source_refs": [],
                        },
                        {
                            "id": "solution",
                            "title": "Solution and results",
                            "content": "The implementation produced a measured result.",
                            "source_refs": [],
                        },
                        {
                            "id": "takeaway",
                            "title": "Takeaway",
                            "content": "The method has a clear bounded use.",
                            "source_refs": [],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "title": "A technical story",
                    "blocks": [
                        {
                            "id": "background",
                            "title": "Background",
                            "content": "The corrected background and constraints.",
                            "source_refs": [],
                        },
                        {
                            "id": "solution",
                            "title": "Solution and results",
                            "content": "The implementation produced a measured result.",
                            "source_refs": [],
                        },
                        {
                            "id": "takeaway",
                            "title": "Takeaway",
                            "content": "The method has a clear bounded use.",
                            "source_refs": [],
                        }
                    ],
                }
            ),
        ]
    )
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    item = make_item()
    item.profile = "tech-blog"
    item.processing.classification.profile = "tech-blog"
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    asyncio.run(enricher._enrich_item(item))

    assert len(requests) == 2
    assert "unknown blocks: section" in requests[1]["user"]
    assert item.processing.artifacts["en"].blocks[0].id == "background"


def test_failed_reenrichment_removes_stale_target_artifact():
    async def complete(**kwargs):
        return json.dumps(
            {
                "title": "A technical story",
                "blocks": [
                    {
                        "id": "story",
                        "type": "section",
                        "title": "",
                        "content": "",
                        "source_refs": [],
                    }
                ],
            }
        )

    item = make_item()
    item.profile = "tech-blog"
    item.processing.classification.profile = "tech-blog"
    item.processing.artifacts["en"] = ContentArtifact(
        language="en",
        title="Stale story",
    )
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    with pytest.raises(ValueError, match="Invalid enrichment artifact"):
        asyncio.run(enricher._enrich_item(item))

    assert "en" not in item.processing.artifacts


def test_enrichment_rejects_cross_block_source_reference():
    block = ContentBlock(
        id="summary",
        type="section",
        title="News",
        content="Content",
        source_refs=["tool-1-1"],
    )
    tool_result = ToolResult(
        request_id="tool-1",
        block_id="background",
        tool="web_search",
        results=[
            {
                "title": "Source",
                "url": "https://example.com/source",
                "text": "Context",
            }
        ],
    )

    with pytest.raises(ValueError, match="unknown source refs"):
        ContentEnricher._validate_blocks(
            [block], PROFILES.get("tech-news"), [tool_result]
        )


def test_enrichment_rejects_empty_required_block():
    block = ContentBlock(
        id="summary",
        type="section",
        title=" ",
        content="",
        source_refs=[],
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        ContentEnricher._validate_blocks(
            [block], PROFILES.get("tech-news"), []
        )


def test_enrichment_batch_reports_failure_without_discarding_successes():
    async def complete(**kwargs):
        raise RuntimeError("AI unavailable")

    successful_item = make_item()
    failed_item = make_item().model_copy(update={"id": "rss:test:failed"})
    enricher = ContentEnricher(
        SimpleNamespace(complete=complete),
        PROFILES,
        ["zh"],
        tools=FakeTools(),
    )

    calls = []

    async def enrich_item(item):  # type: ignore[no-untyped-def]
        calls.append(item.id)
        if item.id == failed_item.id:
            raise RuntimeError("AI unavailable")

    enricher._enrich_item = enrich_item  # type: ignore[method-assign]

    result = asyncio.run(enricher.enrich_batch([successful_item, failed_item]))

    assert result.status == "partial_failure"
    assert result.succeeded_ids == [successful_item.id]
    assert result.failed_ids == [failed_item.id]
    assert result.failures[failed_item.id] == "RuntimeError: AI unavailable"
    assert calls.count(successful_item.id) == 1
    assert calls.count(failed_item.id) == 2


def test_enrichment_batch_retries_item_until_it_succeeds():
    item = make_item()
    calls = 0

    enricher = ContentEnricher(
        SimpleNamespace(complete=lambda **kwargs: None),
        PROFILES,
        ["en"],
        tools=FakeTools(),
    )

    async def enrich_item(_item):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")

    enricher._enrich_item = enrich_item  # type: ignore[method-assign]

    result = asyncio.run(enricher.enrich_batch([item]))

    assert result.status == "success"
    assert calls == 2
