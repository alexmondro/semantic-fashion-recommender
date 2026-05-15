"""Thin LLM client boundary for structured parsing and explanations."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from openai import BadRequestError, NotFoundError, OpenAI, PermissionDeniedError
from pydantic import BaseModel


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
TEXT_COMPLETION_MAX_OUTPUT_TOKENS = 200
MODEL_ACCESS_HINTS = (
    "access",
    "available",
    "does not exist",
    "invalid model",
    "not found",
    "permission",
    "unsupported",
)


class LlmClientError(RuntimeError):
    """Raised when the LLM returns an unusable structured response."""


class LlmClient(Protocol):
    """Minimal interface used by query parsing and explanation modules."""

    def parse_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """Return a Pydantic model parsed from a structured LLM response."""

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return short free-form text for shopper-facing explanations."""


class OpenAiLlmClient:
    """OpenAI Responses API implementation of the LLM boundary."""

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.client = client or OpenAI(api_key=api_key)

    def parse_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """Return a typed object from gpt-5-mini structured outputs."""

        try:
            return self._parse_with_model(
                self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        except (NotFoundError, PermissionDeniedError, BadRequestError) as exc:
            if self.fallback_model is None or not _should_try_fallback(exc):
                raise
            return self._parse_with_model(
                self.fallback_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return one concise shopper-facing sentence."""

        try:
            return self._complete_with_model(
                self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                include_reasoning=True,
            )
        except (NotFoundError, PermissionDeniedError, BadRequestError) as exc:
            if self.fallback_model is None or not _should_try_fallback(exc):
                raise
            return self._complete_with_model(
                self.fallback_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                include_reasoning=False,
            )

    def _parse_with_model(
        self,
        model: str,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        response = self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return parsed

        refusal = _extract_refusal(response)
        if refusal:
            raise LlmClientError(f"OpenAI refused the structured response: {refusal}")
        raise LlmClientError(f"OpenAI response did not include parsed output: {_response_summary(response)}")

    def _complete_with_model(
        self,
        model: str,
        *,
        system_prompt: str,
        user_prompt: str,
        include_reasoning: bool,
    ) -> str:
        request: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "store": False,
            "max_output_tokens": TEXT_COMPLETION_MAX_OUTPUT_TOKENS,
        }
        if include_reasoning:
            request["reasoning"] = {"effort": "minimal"}

        response = self.client.responses.create(**request)
        text = _clean_text(_field(response, "output_text", None))
        if text:
            return text

        refusal = _extract_refusal(response)
        if refusal:
            raise LlmClientError(f"OpenAI refused the text response: {refusal}")
        raise LlmClientError(f"OpenAI response did not include output text: {_response_summary(response)}")


def _should_try_fallback(exc: Exception) -> bool:
    if isinstance(exc, (NotFoundError, PermissionDeniedError)):
        return True
    message = f"{exc} {getattr(exc, 'body', '')}".lower()
    return (
        isinstance(exc, BadRequestError)
        and "model" in message
        and any(hint in message for hint in MODEL_ACCESS_HINTS)
    )


def _extract_refusal(response: Any) -> str | None:
    for output in _iter_items(_field(response, "output", [])):
        for content in _iter_items(_field(output, "content", [])):
            if _field(content, "type") == "refusal":
                refusal = _field(content, "refusal")
                return str(refusal) if refusal else None
    return None


def _response_summary(response: Any) -> str:
    status = _field(response, "status", None)
    output = repr(_field(response, "output", None))
    if len(output) > 1000:
        output = f"{output[:1000]}..."
    return f"status={status!r}, output={output}"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _iter_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
