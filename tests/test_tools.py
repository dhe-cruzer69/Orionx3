from __future__ import annotations

import pytest

import tools


def test_chat_raises_typed_error_with_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(tools.ollama, "chat", unavailable)

    with pytest.raises(tools.LLMProviderError) as exc_info:
        tools._chat("system", "user")

    assert isinstance(exc_info.value.__cause__, OSError)


def test_generation_does_not_turn_provider_failure_into_campaign_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(tools.ollama, "chat", unavailable)

    with pytest.raises(tools.LLMProviderError):
        tools.idea_generator("brief", {})
