"""Selection policy for Horizon's low-priority news brief."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Hashable, Mapping

from ..ai.client import AIClient
from ..ai.localization import normalize_language
from ..ai.markdown import escape_markdown, pangu, safe_url
from ..ai.secondary_brief import (
    SecondaryBriefGenerator,
    SecondaryBriefText,
    fallback_secondary_text,
)
from ..models import ContentItem


TopicDeduplicator = Callable[..., Awaitable[list[ContentItem]]]
UrlKey = Callable[[str], Hashable]


@dataclass
class SecondaryBriefSelection:
    """Items selected for the secondary brief and duplicate diagnostics."""

    items: list[ContentItem]
    duplicate_ids: list[str] = field(default_factory=list)


class SecondaryBriefSelector:
    """Select low-priority items without changing the main digest."""

    def __init__(
        self,
        *,
        config: Any,
        default_profile: str,
        topic_deduplicator: TopicDeduplicator,
        url_key: UrlKey,
    ) -> None:
        self.config = config
        self.default_profile = default_profile
        self.topic_deduplicator = topic_deduplicator
        self.url_key = url_key

    async def select(
        self,
        items: list[ContentItem],
        main_items: list[ContentItem],
        *,
        threshold: float | None = None,
        topic_dedup: bool = True,
        log: bool = True,
    ) -> SecondaryBriefSelection:
        """Return scored items below the main threshold and above the floor."""
        min_score = self.config.digest.secondary_brief.min_score
        main_ids = {item.id for item in main_items}
        main_urls = {self.url_key(str(item.url)) for item in main_items}
        candidates: list[ContentItem] = []

        for item in items:
            if item.id in main_ids or self.url_key(str(item.url)) in main_urls:
                continue
            if not item.processing or not item.processing.analysis:
                continue

            profile_id = self._profile_id(item)
            settings = self.config.processing.profile_settings.get(profile_id)
            effective_threshold = threshold
            if effective_threshold is None and settings is not None:
                effective_threshold = settings.threshold
            score = item.processing.analysis.score
            if (
                effective_threshold is None
                or score is None
                or score >= effective_threshold
                or score < min_score
            ):
                continue
            candidates.append(item)

        candidates.sort(key=self._score, reverse=True)
        if not candidates or not topic_dedup:
            return SecondaryBriefSelection(items=candidates)

        kept_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        candidate_ids = {item.id for item in candidates}
        profile_groups: dict[str, list[ContentItem]] = {}
        for item in [*main_items, *candidates]:
            profile_groups.setdefault(self._profile_id(item), []).append(item)

        for profile_id, profile_items in profile_groups.items():
            settings = self.config.processing.profile_settings.get(profile_id)
            if settings is not None and not settings.topic_dedup:
                kept_ids.update(item.id for item in profile_items)
                continue

            ordered = sorted(profile_items, key=self._score, reverse=True)
            copies = [item.model_copy(deep=True) for item in ordered]
            deduped = await self.topic_deduplicator(copies, log=log)
            deduped_ids = {item.id for item in deduped}
            kept_ids.update(deduped_ids)
            duplicate_ids.update(
                item.id
                for item in profile_items
                if item.id in candidate_ids and item.id not in deduped_ids
            )

        return SecondaryBriefSelection(
            items=[item for item in candidates if item.id in kept_ids],
            duplicate_ids=sorted(duplicate_ids),
        )

    def _profile_id(self, item: ContentItem) -> str:
        if item.processing and item.processing.classification:
            return item.processing.classification.profile
        return self.default_profile

    @staticmethod
    def _score(item: ContentItem) -> float:
        analysis = item.processing.analysis if item.processing else None
        return (
            float(analysis.score)
            if analysis and analysis.score is not None
            else -1.0
        )


class SecondaryBriefRenderer:
    """Render the secondary brief independently from the main digest."""

    _LABELS = {
        "en": {
            "header": "Additional Brief",
            "intro": "These items did not meet the main brief's importance threshold.",
            "empty": "No items met the minimum score for the additional brief.",
        },
        "zh": {
            "header": "补充资讯",
            "intro": "以下内容未达到主简报的重要性阈值，仅作快速浏览。",
            "empty": "今日没有符合补充简报最低分数的新闻。",
        },
    }

    def render(
        self,
        items: list[ContentItem],
        texts: Mapping[str, SecondaryBriefText],
        date: str,
        min_score: float,
        language: str,
        *,
        standalone: bool = True,
    ) -> str:
        """Render concise, source-linked Markdown for selected items."""
        labels = self._LABELS.get(language, self._LABELS["en"])
        if standalone:
            heading = (
                f"# {'Horizon 每日补充资讯' if language == 'zh' else 'Horizon Additional Brief'}"
                f" - {date}"
            )
            if language == "zh":
                intro_text = (
                    f"从 {len(items)} 条低优先级内容中生成补充资讯，"
                    f"最低分数为 {min_score:.1f}。"
                )
            else:
                intro_text = (
                    "A concise brief of lower-priority items "
                    f"(minimum score: {min_score:.1f})."
                )
            intro = f"> {intro_text}\n\n"
        else:
            heading = f"## {escape_markdown(labels['header'])}"
            intro = f"> {escape_markdown(labels['intro'])}\n\n"

        if not items:
            return normalize_language(
                f"{heading}\n\n{escape_markdown(labels['empty'])}\n",
                language,
            )

        lines = [heading, "", intro.rstrip(), ""]
        for item in items:
            analysis = item.processing.analysis if item.processing else None
            score = analysis.score if analysis and analysis.score is not None else "?"
            text = texts.get(item.id)
            title_value = text.title if text else item.title
            summary_value = (
                text.summary
                if text
                else analysis.summary
                if analysis and analysis.summary
                else item.title
            )
            title = escape_markdown(title_value)
            summary = escape_markdown(summary_value)
            url = safe_url(str(item.url))
            title_link = f"[{title}]({url})" if url else title
            score_value = (
                f"{score:.1f}"
                if isinstance(score, (int, float))
                else str(score)
            )
            lines.extend(
                [
                    f"- {title_link} ⭐️ {score_value}/10",
                    f"  {summary}",
                    "",
                ]
            )

        rendered = "\n".join(lines).rstrip() + "\n"
        if language == "zh":
            return normalize_language(pangu(rendered), language)
        return rendered


@dataclass(frozen=True)
class SecondaryBriefOutput:
    """Rendered secondary brief and the saved standalone artifact."""

    report: str
    summary: str
    path: Path | None


class SecondaryBriefService:
    """Generate and persist a secondary brief without changing main rendering."""

    def __init__(self, *, config: Any, storage: Any | None, client: AIClient | None):
        self.config = config
        self.storage = storage
        self.generator = SecondaryBriefGenerator(client) if client else None
        self.renderer = SecondaryBriefRenderer()

    async def build(
        self,
        items: list[ContentItem],
        main_summary: str,
        date: str,
        *,
        language: str,
    ) -> SecondaryBriefOutput:
        generated = (
            await self.generator.generate(items, language=language)
            if self.generator
            else {}
        )
        texts = {
            item.id: generated.get(item.id, fallback_secondary_text(item))
            for item in items
        }
        min_score = self.config.digest.secondary_brief.min_score
        secondary_summary = self.renderer.render(
            items,
            texts,
            date,
            min_score,
            language=language,
            standalone=True,
        )

        report = main_summary
        if self.config.digest.secondary_brief.append_to_main:
            secondary_section = self.renderer.render(
                items,
                texts,
                date,
                min_score,
                language=language,
                standalone=False,
            )
            report = main_summary.rstrip() + "\n\n" + secondary_section.strip() + "\n"

        path = (
            self.storage.save_secondary_summary(
                date,
                secondary_summary,
                language=language,
            )
            if self.storage is not None
            else None
        )
        return SecondaryBriefOutput(
            report=report,
            summary=secondary_summary,
            path=path,
        )
