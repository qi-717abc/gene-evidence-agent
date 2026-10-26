"""Typed, source-aware errors raised by evidence-source adapters."""

from __future__ import annotations

import re
from typing import ClassVar

from gene_evidence_agent.schemas import ToolStatus


class ToolError(Exception):
    """Base error with a stable source identifier and retry hint."""

    status: ClassVar[ToolStatus] = ToolStatus.UPSTREAM_ERROR

    def __init__(self, message: str, *, source: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.source = source
        self.retryable = retryable


class InvalidInputError(ToolError):
    """The adapter rejected malformed or unsupported query input."""

    status = ToolStatus.INVALID_INPUT


class RateLimitedError(ToolError):
    """The upstream source rejected a request because of rate limits."""

    status = ToolStatus.RATE_LIMITED

    def __init__(
        self,
        message: str,
        *,
        source: str,
        retry_after_seconds: float | None = None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message, source=source, retryable=True)
        if retry_after_seconds is not None and retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        if provider_code is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", provider_code):
            raise ValueError("provider_code must be a short, non-sensitive code token")
        self.retry_after_seconds = retry_after_seconds
        self.provider_code = provider_code.lower() if provider_code is not None else None


class ToolTimeoutError(ToolError):
    """An upstream request exceeded its configured time budget."""

    status = ToolStatus.TIMEOUT

    def __init__(self, message: str, *, source: str) -> None:
        super().__init__(message, source=source, retryable=True)


class UpstreamError(ToolError):
    """An upstream source failed or returned an unusable response."""

    status = ToolStatus.UPSTREAM_ERROR

    def __init__(self, message: str, *, source: str, retryable: bool = True) -> None:
        super().__init__(message, source=source, retryable=retryable)


class ModelCallError(UpstreamError):
    """A model API call failed in a safe, classifiable way."""

    FAILURE_CODES = frozenset(
        {
            "transport_error",
            "http_error",
            "invalid_api_response",
            "invalid_json_response",
            "empty_choices",
            "invalid_choice",
            "missing_message",
            "non_text_content",
            "empty_content",
            "invalid_usage",
            "input_token_preflight_mismatch",
            "output_token_limit_exceeded",
        }
    )

    def __init__(
        self,
        message: str,
        *,
        source: str,
        failure_code: str,
        status_code: int | None = None,
        finish_reason: str | None = None,
        reasoning_content_present: bool | None = None,
        retryable: bool = True,
    ) -> None:
        if failure_code not in self.FAILURE_CODES:
            raise ValueError("unsupported model-call failure code")
        if status_code is not None and not 100 <= status_code <= 599:
            raise ValueError("HTTP status code must be between 100 and 599")
        if finish_reason is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", finish_reason):
            raise ValueError("finish_reason must be a short, non-sensitive token")
        if reasoning_content_present is not None and not isinstance(reasoning_content_present, bool):
            raise ValueError("reasoning_content_present must be a boolean or None")
        super().__init__(message, source=source, retryable=retryable)
        self.failure_code = failure_code
        self.status_code = status_code
        self.finish_reason = finish_reason.lower() if finish_reason is not None else None
        # Keep only presence metadata; never retain provider reasoning text.
        self.reasoning_content_present = reasoning_content_present


class ModelOutputError(ModelCallError):
    """A model returned content that cannot safely satisfy a structured-output contract."""

    FAILURE_CODES = frozenset(
        {
            "output_truncated",
            "invalid_json",
            "invalid_schema",
            "unknown_evidence_id",
            "translation_integrity_failure",
        }
    )

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        source: str = "summarizer",
        finish_reason: str | None = None,
        diagnostic_code: str | None = None,
        unmatched_latin_term_count: int | None = None,
    ) -> None:
        if failure_code not in self.FAILURE_CODES:
            raise ValueError("unsupported model-output failure code")
        if diagnostic_code is not None and not re.fullmatch(r"[a-z0-9_]{1,48}", diagnostic_code):
            raise ValueError("diagnostic_code must be a short lowercase code token")
        if unmatched_latin_term_count is not None and (
            isinstance(unmatched_latin_term_count, bool)
            or not isinstance(unmatched_latin_term_count, int)
            or not 1 <= unmatched_latin_term_count <= 1000
        ):
            raise ValueError("unmatched_latin_term_count must be an integer from 1 to 1000")
        super().__init__(
            message,
            source=source,
            failure_code=failure_code,
            finish_reason=finish_reason,
            retryable=False,
        )
        self.diagnostic_code = diagnostic_code
        self.unmatched_latin_term_count = unmatched_latin_term_count
