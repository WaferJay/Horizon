"""Shared AI utility functions."""

import json
import re
from typing import Any, Optional


_MISSING = object()


def _format_json_error(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return (
            f"{error.msg} at line {error.lineno}, column {error.colno} "
            f"(character {error.pos})"
        )
    return str(error) or type(error).__name__


def parse_json_response_with_error(
    response: str,
) -> tuple[Any, str]:
    """Parse a model response and return a useful diagnostic on failure.

    The parser intentionally keeps the permissive extraction strategies used by
    :func:`parse_json_response`, but preserves the first concrete parse failure
    so callers can ask a model to correct its output with actionable feedback.
    ``None`` is a valid JSON value, so an internal sentinel distinguishes it
    from a failed parse.
    """
    text = response.strip()
    if not text:
        return None, "the response was empty"

    errors: list[str] = []

    def try_parse(candidate: str, strategy: str) -> Any:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{strategy}: {_format_json_error(exc)}")
            return _MISSING

    parsed = try_parse(text, "direct JSON")
    if parsed is not _MISSING:
        return parsed, ""

    # Strategy 2: extract from ```json ... ``` code block
    if "```json" in text:
        json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
        parsed = try_parse(json_str, "json code block")
        if parsed is not _MISSING:
            return parsed, ""

    # Strategy 3: extract from ``` ... ``` code block
    if "```" in text:
        json_str = text.split("```", 1)[1].split("```", 1)[0].strip()
        parsed = try_parse(json_str, "code block")
        if parsed is not _MISSING:
            return parsed, ""

    # Strategy 4: find the first { ... } block using brace matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    parsed = try_parse(text[start : i + 1], "embedded JSON object")
                    if parsed is not _MISSING:
                        return parsed, ""
                    break
        else:
            errors.append("embedded JSON object: no matching closing brace")

    # Strategy 5: regex extraction as last resort
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        parsed = try_parse(match.group(), "regex-extracted JSON object")
        if parsed is not _MISSING:
            return parsed, ""

    detail = errors[0] if errors else "no JSON object was found"
    return None, f"could not parse a valid JSON response ({detail})"


def parse_json_response(response: str) -> Optional[dict]:
    """Try multiple strategies to extract a JSON object from an AI response.

    Returns the parsed dict, or None if all strategies fail.
    """
    parsed, _ = parse_json_response_with_error(response)
    return parsed
