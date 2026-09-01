"""Tests for stage-specific provider request extensions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ai.client import OpenAIClient, _create_chained_client
from src.models import (
    AIConfig,
    AIProvider,
    AIStage,
    ClassificationResult,
    ContentAnalysis,
    ContentItem,
    ProcessingResult,
    SourceType,
)
from src.orchestrator import HorizonOrchestrator


def _config(**overrides) -> AIConfig:
    values = {
        "provider": AIProvider.OPENAI,
        "model": "local-model",
        "api_key_env": "OPENAI_API_KEY",
    }
    values.update(overrides)
    return AIConfig(**values)


def _response(content: str = '{"duplicates": []}') -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )


def test_stage_options_are_empty_when_omitted() -> None:
    assert _config().stage_options == {}


def test_stage_options_accept_arbitrary_json_values() -> None:
    config = _config(
        stage_options={
            "topic_dedup": {
                "extra_body": {
                    "repeat_penalty": 1.08,
                    "repeat_last_n": 512,
                    "samplers": ["top_k", "temperature"],
                    "metadata": {"enabled": True, "label": None},
                }
            }
        }
    )

    assert config.stage_options[AIStage.TOPIC_DEDUP].extra_body == {
        "repeat_penalty": 1.08,
        "repeat_last_n": 512,
        "samplers": ["top_k", "temperature"],
        "metadata": {"enabled": True, "label": None},
    }


@pytest.mark.parametrize(
    "stage_options",
    [
        {"dedup": {"extra_body": {"repeat_penalty": 1.08}}},
        {"topic_dedup": {"extra_body": {"model": "override"}}},
        {"topic_dedup": {"extra_body": {"temperature": 0}}},
        {"topic_dedup": {"extra_body": {"custom": float("nan")}}},
    ],
)
def test_stage_options_reject_invalid_configuration(stage_options) -> None:
    with pytest.raises(ValidationError):
        _config(stage_options=stage_options)


@pytest.mark.parametrize(
    "provider,base_url",
    [
        (AIProvider.ANTHROPIC, None),
        (AIProvider.GEMINI, None),
        (AIProvider.MINIMAX, "https://api.minimax.io/anthropic"),
    ],
)
def test_native_clients_reject_stage_options(provider, base_url) -> None:
    with pytest.raises(
        ValidationError,
        match="only supported by OpenAI-compatible providers",
    ):
        _config(
            provider=provider,
            base_url=base_url,
            stage_options={"analysis": {"extra_body": {"custom": True}}},
        )


def test_openai_client_sends_only_the_selected_stage_body(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient(
        _config(
            stage_options={
                "topic_dedup": {
                    "extra_body": {
                        "repeat_penalty": 1.08,
                        "repeat_last_n": 512,
                    }
                },
                "analysis": {"extra_body": {"min_p": 0.05}},
            }
        )
    )

    with patch.object(
        client.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_response(),
    ) as mock_create:
        asyncio.run(
            client.complete(
                system="system",
                user="items",
                stage=AIStage.TOPIC_DEDUP,
            )
        )
        asyncio.run(client.complete(system="system", user="items"))

    assert mock_create.call_args_list[0].kwargs["extra_body"] == {
        "repeat_penalty": 1.08,
        "repeat_last_n": 512,
    }
    assert "extra_body" not in mock_create.call_args_list[1].kwargs


def test_openai_client_preserves_stage_body_during_parameter_retry(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient(
        _config(
            stage_options={
                "topic_dedup": {"extra_body": {"repeat_penalty": 1.08}}
            }
        )
    )

    with patch.object(
        client.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = [
            RuntimeError("temperature is unsupported for this model"),
            _response(),
        ]
        asyncio.run(
            client.complete(
                system="system",
                user="items",
                stage=AIStage.TOPIC_DEDUP,
            )
        )

    assert len(mock_create.call_args_list) == 2
    assert all(
        call.kwargs["extra_body"] == {"repeat_penalty": 1.08}
        for call in mock_create.call_args_list
    )


def test_topic_dedup_marks_its_ai_request_stage() -> None:
    calls = []

    class FakeClient:
        async def complete(self, **kwargs):
            calls.append(kwargs)
            return '{"duplicates": []}'

    items = [
        ContentItem(
            id=f"item-{index}",
            source_type=SourceType.RSS,
            title=f"Item {index}",
            url=f"https://example.com/{index}",
            published_at=datetime.now(timezone.utc),
            processing=ProcessingResult(
                classification=ClassificationResult(
                    profile="tech-news",
                    method="source_override",
                ),
                analysis=ContentAnalysis(
                    score=8 - index,
                    reason="test",
                    summary=f"Summary {index}",
                ),
            ),
        )
        for index in range(2)
    ]
    orchestrator = HorizonOrchestrator.__new__(HorizonOrchestrator)
    orchestrator.console = SimpleNamespace(print=lambda *args, **kwargs: None)
    orchestrator._create_ai_client = lambda: FakeClient()

    result = asyncio.run(orchestrator.merge_topic_duplicates(items, log=False))

    assert result is items
    assert calls[0]["stage"] == AIStage.TOPIC_DEDUP


def test_provider_chain_binds_stage_options_to_configured_provider() -> None:
    config = _config(
        provider_chain="ollama,openai,gemini",
        stage_options={
            "topic_dedup": {"extra_body": {"repeat_penalty": 1.08}}
        },
    )
    chained = _create_chained_client(config)

    assert chained.configs[0].stage_options == {}
    assert chained.configs[1].stage_options == config.stage_options
    assert chained.configs[1].stage_options[AIStage.TOPIC_DEDUP].extra_body == {
        "repeat_penalty": 1.08
    }
    assert chained.configs[2].stage_options == {}
