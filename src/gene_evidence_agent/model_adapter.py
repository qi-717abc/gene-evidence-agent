"""Small OpenAI-compatible chat adapter with explicit opt-in and bounded output."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable
from uuid import uuid4

import httpx

from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import (
    InvalidInputError,
    ModelCallError,
    RateLimitedError,
    ToolTimeoutError,
    UpstreamError,
)
from gene_evidence_agent.http_client import create_async_client
from gene_evidence_agent.model_usage import (
    MODEL_LIFETIME_TOKEN_BUDGET,
    MODEL_SINGLE_REQUEST_TOKEN_BUDGET,
    MODEL_STAGE_TOKEN_BUDGET,
    ModelUsageLedger,
)
from gene_evidence_agent.model_tokenizer import ModelTokenizerError, estimate_prompt_input_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMessage:
    """A text-only chat message; tool execution is deliberately outside this adapter."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ModelResponse:
    """Model text and optional provider-reported usage metadata."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    request_id: str
    finish_reason: str | None = None
    reasoning_content_present: bool | None = None


class _InvalidModelResponse(Exception):
    """Internal safe classification for malformed 2xx responses."""

    def __init__(self, failure_code: str) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code


@runtime_checkable
class ModelAdapter(Protocol):
    """Provider-independent interface used by later planner and summarizer stages."""

    async def complete(
        self,
        messages: Sequence[ModelMessage],
        *,
        max_output_tokens: int | None = None,
        thinking_mode: Literal["enabled", "disabled"] | None = None,
    ) -> ModelResponse: ...


class OpenAICompatibleModelAdapter:
    """Call a `/v1/chat/completions`-compatible service without streaming."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        max_lifetime_tokens: int | None = None,
        allowed_uncertain_requests: int | None = None,
        budget_scope: str = "general",
        max_scope_tokens: int | None = None,
        max_request_tokens: int | None = None,
        max_scope_requests: int | None = None,
        input_token_reserve_margin: int = 0,
        request_stage: str = "unspecified",
        json_mode: bool = False,
    ) -> None:
        if isinstance(input_token_reserve_margin, bool) or not isinstance(input_token_reserve_margin, int) or not 0 <= input_token_reserve_margin <= 4096:
            raise ValueError("input_token_reserve_margin must be an integer between 0 and 4096")
        self._input_token_reserve_margin = input_token_reserve_margin
        self._settings = settings or get_settings()
        self._request_stage = request_stage
        self._json_mode = json_mode
        model_timeout = self._settings.model_request_timeout_seconds
        self._client = client or create_async_client(
            self._settings,
            timeout_seconds=model_timeout,
            connect_timeout_seconds=min(30.0, model_timeout),
        )
        self._owns_client = client is None
        self._usage_ledger = ModelUsageLedger(self._settings.model_usage_db_path)
        self._max_lifetime_tokens = min(
            MODEL_LIFETIME_TOKEN_BUDGET,
            MODEL_LIFETIME_TOKEN_BUDGET if max_lifetime_tokens is None else max_lifetime_tokens,
        )
        self._allowed_uncertain_requests = (
            self._settings.model_allowed_uncertain_requests
            if allowed_uncertain_requests is None
            else allowed_uncertain_requests
        )
        self._budget_scope = budget_scope
        self._max_scope_tokens = min(
            MODEL_STAGE_TOKEN_BUDGET,
            MODEL_STAGE_TOKEN_BUDGET if max_scope_tokens is None else max_scope_tokens,
        )
        self._max_request_tokens = min(
            MODEL_SINGLE_REQUEST_TOKEN_BUDGET,
            MODEL_SINGLE_REQUEST_TOKEN_BUDGET if max_request_tokens is None else max_request_tokens,
        )
        self._max_scope_requests = max_scope_requests
        self.last_usage: tuple[int, int, int] | None = None
        self.request_started = False

    async def aclose(self) -> None:
        """Close only the HTTP client created by this adapter."""

        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: Sequence[ModelMessage],
        *,
        max_output_tokens: int | None = None,
        thinking_mode: Literal["enabled", "disabled"] | None = None,
    ) -> ModelResponse:
        self.last_usage = None
        self.request_started = False
        settings = self._settings
        if not settings.model_enabled:
            raise InvalidInputError("Model calls are disabled by configuration", source="model")
        if settings.model_api_key is None:
            raise InvalidInputError("GEA_MODEL_API_KEY is required before a model call", source="model")
        if not settings.model_name or settings.model_base_url is None:
            raise InvalidInputError("Model name and base URL must be configured", source="model")
        if not messages or any(not message.content.strip() for message in messages):
            raise InvalidInputError("At least one non-empty text message is required", source="model")

        token_limit = settings.model_max_output_tokens if max_output_tokens is None else max_output_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or not 1 <= token_limit <= settings.model_max_output_tokens:
            raise InvalidInputError(
                f"max_output_tokens must be between 1 and {settings.model_max_output_tokens}", source="model"
            )

        headers = {"Authorization": f"Bearer {settings.model_api_key.get_secret_value()}"}
        body = {
            "model": settings.model_name,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "max_tokens": token_limit,
            "stream": False,
        }
        if thinking_mode is not None:
            body["thinking"] = {"type": thinking_mode}
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        url = _chat_completions_url(str(settings.model_base_url))
        try:
            input_token_upper_bound = estimate_prompt_input_tokens(settings.model_name, list(messages)) + self._input_token_reserve_margin
        except ModelTokenizerError as error:
            raise InvalidInputError(
                "No verified tokenizer is available for this model request",
                source="model",
            ) from error
        _validate_context_window(
            input_token_upper_bound,
            token_limit,
            settings.model_context_window_tokens,
        )
        reservation = self._usage_ledger.reserve(
            model=settings.model_name,
            input_token_upper_bound=input_token_upper_bound,
            max_output_tokens=token_limit,
            input_price_credits_per_million=settings.model_input_price_credits_per_million_tokens,
            output_price_credits_per_million=settings.model_output_price_credits_per_million_tokens,
            monthly_budget_credits=settings.model_monthly_budget_credits,
            max_lifetime_tokens=self._max_lifetime_tokens,
            allowed_uncertain_requests=self._allowed_uncertain_requests,
            freeze_on_uncertain_usage=settings.model_freeze_on_uncertain_usage,
            budget_scope=self._budget_scope,
            max_scope_tokens=self._max_scope_tokens,
            max_request_tokens=self._max_request_tokens,
            max_scope_requests=self._max_scope_requests,
        )
        started = time.monotonic()
        local_request_id = str(uuid4())
        self.request_started = True
        try:
            response = await self._client.post(url, headers=headers, json=body)
        except asyncio.CancelledError:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            logger.warning(
                "Model API request cancelled local_request_id=%s provider_request_id=unavailable "
                "stage=%s scope=%s http_status=not_received phase=cancelled elapsed_ms=%s",
                local_request_id,
                self._request_stage,
                self._budget_scope,
                _latency_ms(started),
            )
            raise
        except httpx.TimeoutException as error:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            logger.warning(
                "Model API request timed out local_request_id=%s provider_request_id=unavailable "
                "stage=%s scope=%s http_status=not_received phase=%s elapsed_ms=%s",
                local_request_id,
                self._request_stage,
                self._budget_scope,
                type(error).__name__,
                _latency_ms(started),
            )
            raise ToolTimeoutError("Model API request timed out", source="model") from error
        except httpx.HTTPError as error:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            logger.warning(
                "Model API transport failed local_request_id=%s provider_request_id=unavailable "
                "stage=%s scope=%s http_status=not_received phase=%s elapsed_ms=%s",
                local_request_id,
                self._request_stage,
                self._budget_scope,
                type(error).__name__,
                _latency_ms(started),
            )
            raise ModelCallError(
                "Model API transport failed",
                source="model",
                failure_code="transport_error",
            ) from error

        provider_request_id = _safe_request_id(
            response.headers.get("x-request-id") or response.headers.get("request-id")
        )
        logger.info(
            "Model API response local_request_id=%s provider_request_id=%s stage=%s scope=%s "
            "http_status=%s elapsed_ms=%s content_type=%s",
            local_request_id,
            provider_request_id or "unavailable",
            self._request_stage,
            self._budget_scope,
            response.status_code,
            _latency_ms(started),
            _safe_content_type(response.headers.get("content-type")),
        )

        if response.status_code == 429:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            retry_after = _retry_after(response.headers.get("Retry-After"))
            provider_code = _provider_error_code(response)
            raise RateLimitedError(
                "Model API rate limit was reached",
                source="model",
                retry_after_seconds=retry_after,
                provider_code=provider_code,
            )
        if response.is_error:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            logger.warning(
                "Model API returned upstream HTTP error local_request_id=%s provider_request_id=%s "
                "stage=%s scope=%s http_status=%s elapsed_ms=%s",
                local_request_id,
                provider_request_id or "unavailable",
                self._request_stage,
                self._budget_scope,
                response.status_code,
                _latency_ms(started),
            )
            raise ModelCallError(
                "Model API returned an HTTP error",
                source="model",
                failure_code="http_error",
                status_code=response.status_code,
            )
        try:
            (
                text,
                returned_model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                finish_reason,
                reasoning_content_present,
            ) = _parse_chat_completion_response(response, settings.model_name)
        except _InvalidModelResponse as error:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=_latency_ms(started))
            logger.warning(
                "Model API response invalid local_request_id=%s provider_request_id=%s stage=%s "
                "scope=%s http_status=%s failure_code=%s",
                local_request_id,
                provider_request_id or "unavailable",
                self._request_stage,
                self._budget_scope,
                response.status_code,
                error.failure_code,
            )
            raise ModelCallError(
                "Model API returned an invalid chat-completion response",
                source="model",
                failure_code=error.failure_code,
            ) from error

        latency_ms = _latency_ms(started)
        logger.info(
            "Model API completion parsed local_request_id=%s provider_request_id=%s stage=%s "
            "scope=%s finish_reason=%s content_chars=%s prompt_tokens=%s completion_tokens=%s "
            "total_tokens=%s reasoning_content_present=%s",
            local_request_id,
            provider_request_id or "unavailable",
            self._request_stage,
            self._budget_scope,
            finish_reason or "unavailable",
            len(text),
            prompt_tokens if prompt_tokens is not None else "unavailable",
            completion_tokens if completion_tokens is not None else "unavailable",
            total_tokens if total_tokens is not None else "unavailable",
            reasoning_content_present if reasoning_content_present is not None else "unavailable",
        )
        if prompt_tokens is not None and completion_tokens is not None:
            self._usage_ledger.settle_success(
                reservation,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_price_credits_per_million=settings.model_input_price_credits_per_million_tokens,
                output_price_credits_per_million=settings.model_output_price_credits_per_million_tokens,
                latency_ms=latency_ms,
            )
            self.last_usage = (prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
            if prompt_tokens > input_token_upper_bound:
                raise ModelCallError(
                    "Provider-reported prompt usage exceeded the local tokenizer preflight",
                    source="model",
                    failure_code="input_token_preflight_mismatch",
                    retryable=False,
                )
            if completion_tokens > token_limit:
                logger.warning(
                    "Model API exceeded requested output limit local_request_id=%s provider_request_id=%s "
                    "stage=%s scope=%s http_status=%s requested_max_tokens=%s completion_tokens=%s "
                    "reasoning_content_present=%s",
                    local_request_id,
                    provider_request_id or "unavailable",
                    self._request_stage,
                    self._budget_scope,
                    response.status_code,
                    token_limit,
                    completion_tokens,
                    reasoning_content_present if reasoning_content_present is not None else "unavailable",
                )
                raise ModelCallError(
                    "Provider-reported completion usage exceeded the requested output limit",
                    source="model",
                    failure_code="output_token_limit_exceeded",
                    status_code=response.status_code,
                    finish_reason=finish_reason,
                    reasoning_content_present=reasoning_content_present,
                    retryable=False,
                )
        else:
            self._usage_ledger.mark_uncertain(reservation, latency_ms=latency_ms)

        if not text.strip():
            raise ModelCallError(
                "Model API returned empty chat-completion content",
                source="model",
                failure_code="empty_content",
                finish_reason=finish_reason,
                reasoning_content_present=reasoning_content_present,
            )

        return ModelResponse(
            text=text,
            model=returned_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_id=provider_request_id or local_request_id,
            finish_reason=finish_reason,
            reasoning_content_present=reasoning_content_present,
        )


def _safe_request_id(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if len(candidate) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", candidate):
        return None
    return candidate


def _safe_content_type(value: str | None) -> str:
    if not value:
        return "unavailable"
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", media_type) else "other"


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return f"{clean}/chat/completions" if clean.lower().endswith("/v1") else f"{clean}/v1/chat/completions"


def _validate_context_window(
    input_token_upper_bound: int,
    output_token_limit: int,
    context_window_tokens: int,
) -> None:
    if input_token_upper_bound + output_token_limit > context_window_tokens:
        raise InvalidInputError(
            "Model request exceeds the configured context window; reduce the prompt or output limit",
            source="model",
        )


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("usage token counts must be non-negative integers")
    return value


def _latency_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _provider_error_code(response: httpx.Response) -> str | None:
    """Extract only a short, token-shaped error code; never retain its message/body."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    for key in ("code", "type"):
        value = error.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", value):
            return value.lower()
    return None


def _parse_chat_completion_response(
    response: httpx.Response,
    fallback_model: str,
) -> tuple[str, str, int | None, int | None, int | None, str | None, bool | None]:
    try:
        payload = response.json()
    except ValueError as error:
        raise _InvalidModelResponse("invalid_json_response") from error
    if not isinstance(payload, dict):
        raise _InvalidModelResponse("invalid_api_response")
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise _InvalidModelResponse("invalid_api_response")
    if not choices:
        raise _InvalidModelResponse("empty_choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _InvalidModelResponse("invalid_choice")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _InvalidModelResponse("missing_message")
    text = message.get("content")
    if not isinstance(text, str):
        raise _InvalidModelResponse("non_text_content")
    reasoning_content = message.get("reasoning_content")
    reasoning_content_present = (
        bool(reasoning_content.strip()) if isinstance(reasoning_content, str) else None
    )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        finish_reason = None
    returned_model = payload.get("model", fallback_model)
    if not isinstance(returned_model, str) or not returned_model.strip():
        returned_model = fallback_model
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        raise _InvalidModelResponse("invalid_usage")
    try:
        prompt_tokens = _optional_nonnegative_int(usage.get("prompt_tokens"))
        completion_tokens = _optional_nonnegative_int(usage.get("completion_tokens"))
        total_tokens = _optional_nonnegative_int(usage.get("total_tokens"))
    except ValueError as error:
        raise _InvalidModelResponse("invalid_usage") from error
    return (
        text,
        returned_model,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        finish_reason,
        reasoning_content_present,
    )
