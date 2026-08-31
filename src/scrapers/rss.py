"""RSS feed scraper implementation."""

import calendar
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from email.utils import parsedate_to_datetime
import httpx
import feedparser

from .base import BaseScraper
from ..extractors import ExtractorRegistry
from ..models import ContentItem, SourceType, RSSSourceConfig

logger = logging.getLogger(__name__)


class RSSScraper(BaseScraper):
    """Scraper for RSS/Atom feeds."""

    def __init__(
        self,
        sources: List[RSSSourceConfig],
        http_client: httpx.AsyncClient,
        extractors: Optional[ExtractorRegistry] = None,
    ):
        """Initialize RSS scraper.

        Args:
            sources: List of RSS feed configurations
            http_client: Shared async HTTP client
            extractors: Optional registry of content extractors for full article fetching
        """
        super().__init__({"sources": sources}, http_client)
        self._extractors = extractors

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch RSS feed items.

        Args:
            since: Only fetch items published after this time

        Returns:
            List[ContentItem]: Fetched content items
        """
        items = []
        sources = self.config["sources"]

        for source in sources:
            if not source.enabled:
                continue

            feed_items = await self._fetch_feed(source, since)
            items.extend(feed_items)

        return items

    async def _fetch_feed(
        self, source: RSSSourceConfig, since: datetime
    ) -> List[ContentItem]:
        """Fetch items from a single RSS feed.

        Args:
            source: RSS feed configuration
            since: Only fetch items after this time

        Returns:
            List[ContentItem]: Feed content items
        """
        items = []

        try:
            # Expand environment variables in URL (e.g. ${LWN_TOKEN})
            feed_url = re.sub(
                r"\$\{(\w+)\}",
                lambda m: os.environ.get(m.group(1), m.group(0)).strip(),
                str(source.url),
            )

            # Fetch feed content
            response = await self.client.get(feed_url, follow_redirects=True)
            response.raise_for_status()

            # Parse feed
            feed = feedparser.parse(response.text)

            for entry in feed.entries:
                # Parse published date
                published_at = self._parse_date(entry)
                if not published_at or published_at < since:
                    continue

                # Generate unique ID from feed URL and entry ID
                feed_id = str(source.url).split("//")[1].replace("/", "_")
                entry_id = entry.get("id", entry.get("link", ""))
                entry_hash = hashlib.sha256(str(entry_id).encode("utf-8")).hexdigest()[
                    :16
                ]

                # Extract content
                content = self._extract_content(entry)
                entry_url = self._entry_url(entry, feed_url)

                if source.content_extractor and self._extractors:
                    extractor = self._extractors.get(source.content_extractor)
                    if extractor:
                        if entry_url:
                            full = await extractor.extract(entry_url, self.client)
                            if full:
                                content = full

                item = ContentItem(
                    id=self._generate_id("rss", feed_id, entry_hash),
                    source_type=SourceType.RSS,
                    title=entry.get("title", "Untitled"),
                    url=entry_url,
                    content=content,
                    author=entry.get("author", source.name),
                    published_at=published_at,
                    profile=source.profile,
                    metadata={
                        "feed_name": source.name,
                        "category": source.category,
                        "tags": [tag.term for tag in entry.get("tags", [])],
                    },
                )
                items.append(item)

        except httpx.HTTPError as e:
            logger.warning("Error fetching RSS feed %s: %s", source.name, e)
        except Exception as e:
            logger.warning("Error parsing RSS feed %s: %s", source.name, e)

        return items

    def _entry_url(self, entry: dict, feed_url: str) -> str:
        """Return an absolute article URL using Atom link-relation semantics."""
        links = entry.get("links", [])

        # An omitted rel is equivalent to rel="alternate" in Atom. Prefer an
        # HTML representation, but retain document order within each group.
        for relation in ("alternate", "via"):
            candidates = [
                candidate
                for candidate in links
                if str(candidate.get("rel") or "alternate").lower() == relation
            ]
            candidates.sort(key=self._link_media_priority)
            for candidate in candidates:
                resolved = self._resolve_http_link(candidate.get("href"), feed_url)
                if resolved:
                    return resolved

        # RSS parsers and non-conforming feeds may expose only the convenience
        # `link` field without a corresponding relation entry.
        resolved = self._resolve_http_link(entry.get("link"), feed_url)
        return resolved or feed_url

    @staticmethod
    def _link_media_priority(link: dict) -> int:
        """Prefer browser-readable alternate representations."""
        media_type = str(link.get("type") or "").split(";", 1)[0].strip().lower()
        return 0 if media_type in {"text/html", "application/xhtml+xml"} else 1

    @staticmethod
    def _resolve_http_link(href: object, base_url: str) -> str:
        """Resolve an IRI reference and accept only fetchable HTTP(S) URLs."""
        value = str(href or "").strip()
        if not value:
            return ""
        resolved = urljoin(base_url, value)
        return resolved if urlparse(resolved).scheme in {"http", "https"} else ""

    def _parse_date(self, entry: dict) -> datetime:
        """Parse publication date from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            datetime: Parsed publication date or None
        """
        # Try different date fields
        for field in ["published", "updated", "created"]:
            if field in entry:
                try:
                    # Try parsing structured time first
                    if f"{field}_parsed" in entry and entry[f"{field}_parsed"]:
                        return datetime.fromtimestamp(
                            calendar.timegm(entry[f"{field}_parsed"]), tz=timezone.utc
                        )
                    # Fallback to string parsing
                    date_str = entry[field]
                    return parsedate_to_datetime(date_str)
                except Exception:
                    continue

        return None

    def _extract_content(self, entry: dict) -> str:
        """Extract text content from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            str: Extracted text content
        """
        # Try different content fields
        if "summary" in entry:
            return entry.summary
        if "description" in entry:
            return entry.description
        if "content" in entry and entry.content:
            # content is usually a list
            return entry.content[0].get("value", "")

        return ""
