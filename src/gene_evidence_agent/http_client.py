"""Shared HTTP client constructors for external biomedical APIs."""

from __future__ import annotations

import httpx

from gene_evidence_agent.config import Settings, get_settings


def build_http_timeout(
    settings: Settings | None = None,
    *,
    timeout_seconds: float | None = None,
    connect_timeout_seconds: float | None = None,
) -> httpx.Timeout:
    """Build a bounded timeout, optionally overriding the shared source timeout."""

    resolved_settings = settings or get_settings()
    resolved_timeout = (
        resolved_settings.request_timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    return httpx.Timeout(resolved_timeout, connect=connect_timeout_seconds)


def create_async_client(
    settings: Settings | None = None,
    *,
    timeout_seconds: float | None = None,
    connect_timeout_seconds: float | None = None,
) -> httpx.AsyncClient:
    """Create an async client with an optional per-service timeout override."""

    return httpx.AsyncClient(
        timeout=build_http_timeout(
            settings,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
        ),
        follow_redirects=False,
        headers={"User-Agent": "gene-evidence-agent/0.1.0"},
    )

