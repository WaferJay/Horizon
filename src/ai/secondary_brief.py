"""Low-priority news brief generation.

This module is deliberately separate from the main enrichment pipeline.  A
secondary brief only needs the already-produced analysis summary and never
performs web search or full item enrichment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from ..debug.context import debug_scope
from ..models import ContentItem
from .client import AIClient
from .utils import parse_json_response_with_error

logger = logging.getLogger(__name__)

_MAX_BATCH_ITEMS = 20
_MAX_CONTENT_EXCERPT = 1200


@dataclass(frozen=True)
class SecondaryBriefText:
    """Localized text generated for one secondary-brief item."""

    title: str
    summary: str


class _GeneratedItem(BaseModel):
    id: str
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class _GeneratedResponse(BaseModel):
    items: list[_GeneratedItem] = Field(default_factory=list)


class SecondaryBriefGenerator:
    """Generate concise localized summaries in bounded batches."""

    def __init__(self, client: AIClient):
        self.client = client

    async def generate(
        self,
        items: list[ContentItem],
        *,
        language: str,
    ) -> dict[str, SecondaryBriefText]:
        if not items:
            return {}

        generated: dict[str, SecondaryBriefText] = {}
        for batch in self._batches(items):
            generated.update(await self._generate_batch(batch, language=language))
        return generated

    async def _generate_batch(
        self,
        items: list[ContentItem],
        *,
        language: str,
    ) -> dict[str, SecondaryBriefText]:
        user = self._user_prompt(items, language)
        try:
            with debug_scope(stage="secondary_brief", language=language):
                response = await self.client.complete(
                    system=self._system_prompt(language),
                    user=user,
                )
        except Exception as exc:  # pragma: no cover - provider-specific failure
            logger.warning("Secondary brief generation failed: %s", exc)
            return {}

        parsed, error = parse_json_response_with_error(response)
        try:
            result = _GeneratedResponse.model_validate(parsed)
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Secondary brief response was invalid (%s): %s",
                error or "schema validation failed",
                exc,
            )
            return {}

        allowed_ids = {item.id for item in items}
        output: dict[str, SecondaryBriefText] = {}
        for entry in result.items:
            if entry.id not in allowed_ids or entry.id in output:
                continue
            output[entry.id] = SecondaryBriefText(
                title=entry.title.strip(),
                summary=entry.summary.strip(),
            )
        return output

    @staticmethod
    def _batches(items: list[ContentItem]) -> Iterable[list[ContentItem]]:
        for start in range(0, len(items), _MAX_BATCH_ITEMS):
            yield items[start : start + _MAX_BATCH_ITEMS]

    @staticmethod
    def _system_prompt(language: str) -> str:
        return f"""You are a concise news editor creating a low-priority news brief.

Write in language code: {language}.
Use only the supplied title and source summary. Do not add facts, context,
causes, consequences, numbers, or claims that are not supplied. Preserve
uncertainty and attribution. Each item must have one short sentence, preferably
under 60 Chinese characters or 30 words in other languages.

Return valid JSON only in this shape:
{{"items":[{{"id":"<input id>","title":"<short title>","summary":"<one sentence>"}}]}}
Return at most one output entry for each supplied ID."""

    @staticmethod
    def _user_prompt(items: list[ContentItem], language: str) -> str:
        lines = [f"Create a brief in language code: {language}."]
        for item in items:
            analysis = item.processing.analysis if item.processing else None
            source_summary = analysis.summary if analysis else ""
            if not source_summary:
                source_summary = (item.content or "").strip()[:_MAX_CONTENT_EXCERPT]
            lines.extend(
                [
                    f"\nID: {item.id}",
                    f"Title: {item.title}",
                    "Score: "
                    f"{analysis.score if analysis and analysis.score is not None else '?'} / 10",
                    f"Source summary: {source_summary or item.title}",
                ]
            )
        return "\n".join(lines)


def fallback_secondary_text(item: ContentItem) -> SecondaryBriefText:
    """Return source-backed text when the optional AI pass is unavailable."""
    analysis = item.processing.analysis if item.processing else None
    return SecondaryBriefText(
        title=item.title,
        summary=(analysis.summary if analysis and analysis.summary else item.title),
    )
