"""MCP adapter glue for the secondary-brief pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..services.secondary_brief import SecondaryBriefRenderer
from .horizon_adapter import make_storage


@dataclass
class SecondarySummaryArtifacts:
    """Rendered secondary output and optional Horizon data publication."""

    report: str
    summary: str
    storage: Any | None
    published_path: Any | None


async def select_secondary_items(
    orchestrator: Any,
    items: list[Any],
    main_items: list[Any],
    *,
    threshold: float | None,
    topic_dedup: bool,
) -> tuple[list[Any], list[str]]:
    """Call the optional native selector while supporting older adapters."""
    selector = getattr(orchestrator, "select_secondary_items", None)
    if not callable(selector):
        return [], []

    result = await selector(
        items,
        main_items,
        threshold=threshold,
        topic_dedup=topic_dedup,
        log=False,
    )
    return result.items, list(getattr(result, "duplicate_ids", []))


async def build_secondary_summary(
    ctx: Any,
    items: list[Any],
    main_summary: str,
    date: str,
    *,
    language: str,
    save_to_horizon_data: bool,
) -> SecondarySummaryArtifacts:
    """Use the loaded native secondary service and publish when requested."""
    storage = (
        make_storage(ctx.runtime, ctx.config_path)
        if save_to_horizon_data
        else None
    )
    service_class = getattr(ctx.runtime, "SecondaryBriefService", None)
    if service_class is not None:
        create_ai_client = getattr(ctx.runtime, "create_ai_client", None)
        service = service_class(
            config=ctx.config,
            storage=storage,
            client=(
                create_ai_client(ctx.config.ai)
                if items and callable(create_ai_client)
                else None
            ),
        )
        output = await service.build(
            items,
            main_summary,
            date,
            language=language,
        )
        return SecondarySummaryArtifacts(
            report=output.report,
            summary=output.summary,
            storage=storage,
            published_path=output.path,
        )

    # Compatibility for lightweight runtime adapters that predate this service.
    secondary_config = getattr(
        getattr(ctx.config, "digest", None), "secondary_brief", None
    )
    min_score = getattr(secondary_config, "min_score", 0.0)
    renderer_class = getattr(
        ctx.runtime, "SecondaryBriefRenderer", SecondaryBriefRenderer
    )
    renderer = renderer_class()
    secondary_summary = renderer.render(
        items,
        {},
        date,
        min_score,
        language=language,
        standalone=True,
    )
    report = main_summary
    if getattr(secondary_config, "append_to_main", False):
        secondary_section = renderer.render(
            items,
            {},
            date,
            min_score,
            language=language,
            standalone=False,
        )
        report = main_summary.rstrip() + "\n\n" + secondary_section.strip() + "\n"
    published_path = (
        storage.save_secondary_summary(date, secondary_summary, language=language)
        if storage is not None
        else None
    )
    return SecondarySummaryArtifacts(
        report=report,
        summary=secondary_summary,
        storage=storage,
        published_path=published_path,
    )
