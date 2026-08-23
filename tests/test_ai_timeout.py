"""Tests for AI provider timeout configuration."""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from src.ai.client import (
    AnthropicClient,
    AzureOpenAIClient,
    GeminiClient,
    OpenAIClient,
    _build_http_timeout,
    _create_chained_client,
)
from src.models import AIConfig, AIProvider


def _make_config(provider: AIProvider, **overrides) -> AIConfig:
    defaults = {
        "provider": provider,
        "model": "test-model",
        "api_key_env": "TEST_API_KEY",
    }
    defaults.update(overrides)
    return AIConfig(**defaults)


def test_timeout_fields_are_optional_by_default():
    config = _make_config(AIProvider.OPENAI)

    assert config.connect_timeout_sec is None
    assert config.read_timeout_sec is None
    assert config.write_timeout_sec is None
    assert _build_http_timeout(config) is None


@pytest.mark.parametrize(
    "field",
    ["connect_timeout_sec", "read_timeout_sec", "write_timeout_sec"],
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "bad"])
def test_timeout_fields_require_finite_positive_numbers(field, value):
    with pytest.raises(ValidationError):
        _make_config(AIProvider.OPENAI, **{field: value})


def test_http_timeout_preserves_sdk_defaults_for_unset_phases():
    timeout = _build_http_timeout(
        _make_config(AIProvider.OPENAI, read_timeout_sec=120)
    )

    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 120
    assert timeout.write == 600.0
    assert timeout.pool == 600.0


def test_openai_client_receives_phase_timeouts(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _make_config(
        AIProvider.OPENAI,
        connect_timeout_sec=10,
        read_timeout_sec=300,
        write_timeout_sec=60,
    )

    with patch("src.ai.client.AsyncOpenAI") as sdk_client:
        OpenAIClient(config)

    timeout = sdk_client.call_args.kwargs["timeout"]
    assert timeout.connect == 10
    assert timeout.read == 300
    assert timeout.write == 60


def test_anthropic_client_receives_phase_timeouts(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _make_config(
        AIProvider.ANTHROPIC,
        connect_timeout_sec=10,
        read_timeout_sec=300,
        write_timeout_sec=60,
    )

    with patch("src.ai.client.AsyncAnthropic") as sdk_client:
        AnthropicClient(config)

    timeout = sdk_client.call_args.kwargs["timeout"]
    assert timeout.connect == 10
    assert timeout.read == 300
    assert timeout.write == 60


def test_azure_client_receives_phase_timeouts(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    monkeypatch.setenv("TEST_AZURE_ENDPOINT", "https://example.openai.azure.com")
    config = _make_config(
        AIProvider.AZURE,
        azure_endpoint_env="TEST_AZURE_ENDPOINT",
        api_version="2024-10-21",
        connect_timeout_sec=10,
        read_timeout_sec=300,
        write_timeout_sec=60,
    )

    with patch("src.ai.client.AsyncAzureOpenAI") as sdk_client:
        AzureOpenAIClient(config)

    timeout = sdk_client.call_args.kwargs["timeout"]
    assert timeout.connect == 10
    assert timeout.read == 300
    assert timeout.write == 60


def test_clients_do_not_override_sdk_timeout_when_unconfigured(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    with patch("src.ai.client.AsyncOpenAI") as sdk_client:
        OpenAIClient(_make_config(AIProvider.OPENAI))

    assert "timeout" not in sdk_client.call_args.kwargs


def test_gemini_maps_read_timeout_to_milliseconds(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _make_config(AIProvider.GEMINI, read_timeout_sec=300)

    with patch("src.ai.client.genai.Client") as sdk_client:
        GeminiClient(config)

    options = sdk_client.call_args.kwargs["http_options"]
    assert options.timeout == 300_000


def test_gemini_warns_for_unsupported_connect_and_write_timeouts(
    monkeypatch, caplog
):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = _make_config(
        AIProvider.GEMINI,
        connect_timeout_sec=10,
        write_timeout_sec=60,
    )

    with patch("src.ai.client.genai.Client") as sdk_client:
        with caplog.at_level(logging.WARNING, logger="src.ai.client"):
            GeminiClient(config)

    assert "ai.connect_timeout_sec" in caplog.text
    assert "ai.write_timeout_sec" in caplog.text
    assert "http_options" not in sdk_client.call_args.kwargs


def test_chained_clients_inherit_timeout_configuration():
    config = _make_config(
        AIProvider.OPENAI,
        provider_chain="openai,gemini",
        connect_timeout_sec=10,
        read_timeout_sec=300,
        write_timeout_sec=60,
    )

    chained = _create_chained_client(config)

    assert len(chained.configs) == 2
    for chain_config in chained.configs:
        assert chain_config.connect_timeout_sec == 10
        assert chain_config.read_timeout_sec == 300
        assert chain_config.write_timeout_sec == 60
