"""Focused coverage for the OpenAI LLM boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.llm_client import LlmClientError, OpenAiLlmClient, TEXT_COMPLETION_MAX_OUTPUT_TOKENS


class TinyParsedModel(BaseModel):
    value: str


class FakeParsedResponse:
    def __init__(
        self,
        parsed: BaseModel | None = None,
        *,
        output: list[Any] | None = None,
        status: str = "completed",
    ) -> None:
        self.output_parsed = parsed
        self.output = output or []
        self.status = status


class FakeTextResponse:
    def __init__(
        self,
        output_text: str | None = None,
        *,
        output: list[Any] | None = None,
        status: str = "completed",
    ) -> None:
        self.output_text = output_text
        self.output = output or []
        self.status = status


class FakeResponses:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> FakeParsedResponse:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def create(self, **kwargs: Any) -> FakeTextResponse:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAiClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.responses = FakeResponses(outcomes)


def api_status_error(error_type: type[Exception], message: str, status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request)
    return error_type(message, response=response, body={"error": {"message": message}})


def timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.APITimeoutError(request=request)


def connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.APIConnectionError(request=request)


def test_parse_structured_uses_fallback_model_for_model_not_found() -> None:
    """A missing primary model should retry once with the configured fallback."""

    primary_error = api_status_error(openai.NotFoundError, "model not found", 404)
    fallback_response = FakeParsedResponse(TinyParsedModel(value="ok"))
    fake_client = FakeOpenAiClient([primary_error, fallback_response])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=fake_client,
    )

    parsed = llm.parse_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=TinyParsedModel,
    )

    assert parsed == TinyParsedModel(value="ok")
    assert [call["model"] for call in fake_client.responses.calls] == [
        "gpt-5-mini",
        "gpt-4o-mini",
    ]
    first_call = fake_client.responses.calls[0]
    assert first_call["text_format"] is TinyParsedModel
    assert first_call["input"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


@pytest.mark.parametrize(
    "error",
    [
        api_status_error(openai.RateLimitError, "rate limit", 429),
        timeout_error(),
        connection_error(),
    ],
)
def test_parse_structured_does_not_fallback_for_transient_errors(error: Exception) -> None:
    """Transient errors should surface instead of silently changing models."""

    fake_client = FakeOpenAiClient([error])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=fake_client,
    )

    with pytest.raises(type(error)):
        llm.parse_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=TinyParsedModel,
        )

    assert [call["model"] for call in fake_client.responses.calls] == ["gpt-5-mini"]


def test_parse_structured_propagates_without_fallback_model() -> None:
    """If fallback_model is None, the original model error should propagate."""

    primary_error = api_status_error(openai.NotFoundError, "model not found", 404)
    fake_client = FakeOpenAiClient([primary_error])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model=None,
        client=fake_client,
    )

    with pytest.raises(openai.NotFoundError):
        llm.parse_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=TinyParsedModel,
        )

    assert [call["model"] for call in fake_client.responses.calls] == ["gpt-5-mini"]


def test_parse_structured_refusal_error_includes_refusal_reason() -> None:
    """Refusals should be distinguishable from malformed empty parsed output."""

    refusal_response = FakeParsedResponse(
        None,
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="refusal",
                        refusal="I cannot parse this shopping request.",
                    )
                ]
            )
        ],
    )
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=FakeOpenAiClient([refusal_response]),
    )

    with pytest.raises(LlmClientError, match="I cannot parse this shopping request"):
        llm.parse_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=TinyParsedModel,
        )


def test_complete_text_uses_primary_model_with_reasoning_controls() -> None:
    """Short text completions should constrain storage, reasoning, and output length."""

    fake_client = FakeOpenAiClient([FakeTextResponse("  These sandals fit the beach request.  ")])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=fake_client,
    )

    text = llm.complete_text(system_prompt="system", user_prompt="user")

    assert text == "These sandals fit the beach request."
    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "minimal"}
    assert call["max_output_tokens"] == TEXT_COMPLETION_MAX_OUTPUT_TOKENS
    assert call["input"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_complete_text_fallback_omits_reasoning_controls() -> None:
    """The gpt-4o-mini fallback should not receive reasoning-only kwargs."""

    primary_error = api_status_error(openai.NotFoundError, "model not found", 404)
    fake_client = FakeOpenAiClient([primary_error, FakeTextResponse("Fallback explanation.")])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=fake_client,
    )

    text = llm.complete_text(system_prompt="system", user_prompt="user")

    assert text == "Fallback explanation."
    assert [call["model"] for call in fake_client.responses.calls] == [
        "gpt-5-mini",
        "gpt-4o-mini",
    ]
    primary_call, fallback_call = fake_client.responses.calls
    assert primary_call["reasoning"] == {"effort": "minimal"}
    assert fallback_call["store"] is False
    assert fallback_call["max_output_tokens"] == TEXT_COMPLETION_MAX_OUTPUT_TOKENS
    assert "reasoning" not in fallback_call


@pytest.mark.parametrize(
    "error",
    [
        api_status_error(openai.RateLimitError, "rate limit", 429),
        timeout_error(),
        connection_error(),
    ],
)
def test_complete_text_does_not_fallback_for_transient_errors(error: Exception) -> None:
    """Transient errors should surface instead of silently changing models."""

    fake_client = FakeOpenAiClient([error])
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=fake_client,
    )

    with pytest.raises(type(error)):
        llm.complete_text(system_prompt="system", user_prompt="user")

    assert [call["model"] for call in fake_client.responses.calls] == ["gpt-5-mini"]


def test_complete_text_refusal_error_includes_refusal_reason() -> None:
    """Refusals should be distinguished from blank output_text responses."""

    refusal_response = FakeTextResponse(
        " ",
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="refusal",
                        refusal="I cannot explain this recommendation.",
                    )
                ]
            )
        ],
    )
    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=FakeOpenAiClient([refusal_response]),
    )

    with pytest.raises(LlmClientError, match="I cannot explain this recommendation"):
        llm.complete_text(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "response",
    [
        FakeTextResponse("   "),
        SimpleNamespace(output=[], status="completed"),
    ],
)
def test_complete_text_blank_or_malformed_response_raises(response: Any) -> None:
    """Blank visible text without a refusal is an unusable LLM response."""

    llm = OpenAiLlmClient(
        api_key="test-key",
        model="gpt-5-mini",
        fallback_model="gpt-4o-mini",
        client=FakeOpenAiClient([response]),
    )

    with pytest.raises(LlmClientError, match="did not include output text"):
        llm.complete_text(system_prompt="system", user_prompt="user")
