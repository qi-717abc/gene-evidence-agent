"""Persistent token/cost ledger and monthly model-credit budget gate."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable
from uuid import uuid4

from gene_evidence_agent.errors import InvalidInputError

_CREDITS_PER_MILLION = Decimal("1000000")
_SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
MODEL_LIFETIME_TOKEN_BUDGET = 1_000_000_000
MODEL_STAGE_TOKEN_BUDGET = 10_000_000
MODEL_SINGLE_REQUEST_TOKEN_BUDGET = 10_000_000
MODEL_LOCALIZATION_TOKEN_BUDGET = 10_000_000
P5_LOCALIZATION_BUDGET_SCOPE = "p5-chinese-localization"
MODEL_P6_E2E_TOKEN_BUDGET = 10_000_000
P6_E2E_BUDGET_SCOPE = "p6-web-e2e"
MODEL_REPAIR_TOKEN_BUDGET = 10_000_000
REPAIR_E2E_BUDGET_SCOPE = "repair-e2e"
_P5_LOCALIZATION_BUDGET_SCOPES = (
    P5_LOCALIZATION_BUDGET_SCOPE,
    "cli-chinese-localization",
    "web-p5-localization",
)


@dataclass(frozen=True)
class UsageReservation:
    """Preflight reservation that counts against the current month's budget."""

    request_id: str
    month_key: str
    reserved_cost_credits: Decimal


@dataclass(frozen=True)
class ModelUsageSnapshot:
    """Aggregate usage for a single local calendar month."""

    month_key: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    cost_credits: Decimal
    uncertain_requests: int


@dataclass(frozen=True)
class TokenBudgetSnapshot:
    """Token accounting with uncertain requests conservatively held at their reserve."""

    used_tokens: int
    confirmed_tokens: int
    reserved_tokens: int
    request_count: int
    uncertain_requests: int
    reserved_requests: int


class ModelUsageLedger:
    """SQLite-backed monthly usage tracker; request content and credentials are never stored."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = Path(database_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def reserve(
        self,
        *,
        model: str,
        input_token_upper_bound: int,
        max_output_tokens: int,
        input_price_credits_per_million: Decimal,
        output_price_credits_per_million: Decimal,
        monthly_budget_credits: Decimal,
        max_lifetime_tokens: int | None = None,
        allowed_uncertain_requests: int = 0,
        freeze_on_uncertain_usage: bool = True,
        budget_scope: str = "general",
        max_scope_tokens: int | None = None,
        max_request_tokens: int | None = None,
        max_scope_requests: int | None = None,
    ) -> UsageReservation:
        """Atomically reserve a conservative maximum request cost before network I/O."""

        if input_token_upper_bound < 0 or max_output_tokens < 1:
            raise ValueError("token bounds must be non-negative and output cap must be positive")
        if min(input_price_credits_per_million, output_price_credits_per_million) < 0:
            raise ValueError("token prices must be non-negative")
        if monthly_budget_credits <= 0:
            raise ValueError("monthly budget must be positive")
        if max_lifetime_tokens is not None and max_lifetime_tokens < 1:
            raise ValueError("max_lifetime_tokens must be positive")
        if (
            isinstance(allowed_uncertain_requests, bool)
            or not isinstance(allowed_uncertain_requests, int)
            or allowed_uncertain_requests < 0
        ):
            raise ValueError("allowed_uncertain_requests must be a non-negative integer")
        if max_scope_tokens is not None and max_scope_tokens < 1:
            raise ValueError("max_scope_tokens must be positive")
        if max_request_tokens is not None and max_request_tokens < 1:
            raise ValueError("max_request_tokens must be positive")
        if max_scope_requests is not None and (
            isinstance(max_scope_requests, bool)
            or not isinstance(max_scope_requests, int)
            or max_scope_requests < 1
        ):
            raise ValueError("max_scope_requests must be a positive integer")
        if not budget_scope.strip() or len(budget_scope) > 64:
            raise ValueError("budget_scope must be a short non-empty label")
        request_token_reserve = input_token_upper_bound + max_output_tokens
        if max_request_tokens is not None and request_token_reserve > max_request_tokens:
            raise InvalidInputError(
                "Model request would exceed its token reserve; no model request was sent",
                source="model_budget",
            )

        input_reserve = Decimal(input_token_upper_bound) * input_price_credits_per_million / _CREDITS_PER_MILLION
        output_reserve = Decimal(max_output_tokens) * output_price_credits_per_million / _CREDITS_PER_MILLION
        reserved_cost = input_reserve + output_reserve
        now = self._now()
        month_key = now.astimezone(_SHANGHAI).strftime("%Y-%m")
        request_id = str(uuid4())
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_cost = _monthly_cost(connection, month_key)
            if current_cost + reserved_cost > monthly_budget_credits:
                connection.rollback()
                raise InvalidInputError(
                    "Monthly model-credit budget would be exceeded; no model request was sent",
                    source="model_budget",
                )
            if max_lifetime_tokens is not None:
                lifetime = _token_usage(connection)
                if freeze_on_uncertain_usage and lifetime.uncertain_requests > allowed_uncertain_requests:
                    connection.rollback()
                    raise InvalidInputError(
                        "Unknown model usage exceeds the explicitly allowed historical baseline; "
                        "subsequent model requests are frozen",
                        source="model_budget",
                    )
                if lifetime.used_tokens + request_token_reserve > max_lifetime_tokens:
                    connection.rollback()
                    raise InvalidInputError(
                        "Lifetime model-token budget would be exceeded; no model request was sent",
                        source="model_budget",
                    )
            if max_scope_tokens is not None:
                scoped = _token_usage(connection, budget_scope=budget_scope)
                if scoped.used_tokens + request_token_reserve > max_scope_tokens:
                    connection.rollback()
                    raise InvalidInputError(
                        "Scoped model-token budget would be exceeded; no model request was sent",
                        source="model_budget",
                    )
            if max_scope_requests is not None:
                scoped = _token_usage(connection, budget_scope=budget_scope)
                if scoped.request_count >= max_scope_requests:
                    connection.rollback()
                    raise InvalidInputError(
                        "Scoped model request-count limit was reached; no model request was sent",
                        source="model_budget",
                    )
            connection.execute(
                """INSERT INTO model_usage (
                    request_id, month_key, model, input_token_upper_bound, max_output_tokens,
                    reserved_cost_credits, status, started_at, budget_scope
                ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?)""",
                (
                    request_id,
                    month_key,
                    model,
                    input_token_upper_bound,
                    max_output_tokens,
                    str(reserved_cost),
                    now.isoformat(),
                    budget_scope,
                ),
            )
            connection.commit()
        return UsageReservation(request_id, month_key, reserved_cost)

    def settle_success(
        self,
        reservation: UsageReservation,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        input_price_credits_per_million: Decimal,
        output_price_credits_per_million: Decimal,
        latency_ms: int,
    ) -> Decimal:
        """Record server-reported token usage and replace the conservative reserve."""

        if min(prompt_tokens, completion_tokens, latency_ms) < 0:
            raise ValueError("reported usage and latency must be non-negative")
        actual_cost = (
            Decimal(prompt_tokens) * input_price_credits_per_million
            + Decimal(completion_tokens) * output_price_credits_per_million
        ) / _CREDITS_PER_MILLION
        completed_at = self._now().isoformat()
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE model_usage SET prompt_tokens=?, completion_tokens=?, actual_cost_credits=?,
                    status='success', latency_ms=?, completed_at=?
                    WHERE request_id=? AND status='reserved'""",
                (
                    prompt_tokens,
                    completion_tokens,
                    str(actual_cost),
                    latency_ms,
                    completed_at,
                    reservation.request_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("model usage reservation was missing or already finalized")
            connection.commit()
        return actual_cost

    def mark_uncertain(self, reservation: UsageReservation, *, latency_ms: int) -> None:
        """Keep the full reserve if a request outcome/usage is unknown, preventing overspend."""

        if latency_ms < 0:
            raise ValueError("latency must be non-negative")
        with self._database() as connection:
            connection.execute(
                """UPDATE model_usage SET status='uncertain', latency_ms=?, completed_at=?
                    WHERE request_id=? AND status='reserved'""",
                (latency_ms, self._now().isoformat(), reservation.request_id),
            )

    def snapshot(self, *, month_key: str | None = None) -> ModelUsageSnapshot:
        """Return a month aggregate without exposing queries, prompts, or API credentials."""

        month = month_key or self._now().astimezone(_SHANGHAI).strftime("%Y-%m")
        with self._database() as connection:
            rows = connection.execute(
                """SELECT status, prompt_tokens, completion_tokens, reserved_cost_credits,
                    actual_cost_credits FROM model_usage WHERE month_key=?""",
                (month,),
            ).fetchall()
        prompt_tokens = sum(row[1] or 0 for row in rows)
        completion_tokens = sum(row[2] or 0 for row in rows)
        cost = sum(
            (Decimal(row[4]) if row[0] == "success" else Decimal(row[3]) for row in rows),
            start=Decimal("0"),
        )
        return ModelUsageSnapshot(
            month_key=month,
            request_count=len(rows),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_credits=cost,
            uncertain_requests=sum(row[0] == "uncertain" for row in rows),
        )

    def token_budget_snapshot(self, *, budget_scope: str | None = None) -> TokenBudgetSnapshot:
        """Return lifetime token use; uncertain and in-flight requests count at full reserve."""

        with self._database() as connection:
            usage = _token_usage(connection, budget_scope=budget_scope)
        return usage

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        return value

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=15)
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS model_usage (
                request_id TEXT PRIMARY KEY,
                month_key TEXT NOT NULL,
                model TEXT NOT NULL,
                input_token_upper_bound INTEGER NOT NULL,
                max_output_tokens INTEGER NOT NULL,
                reserved_cost_credits TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                actual_cost_credits TEXT,
                status TEXT NOT NULL CHECK(status IN ('reserved', 'success', 'uncertain')),
                latency_ms INTEGER,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                budget_scope TEXT NOT NULL DEFAULT 'general'
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_usage)")}
        if "budget_scope" not in columns:
            connection.execute(
                "ALTER TABLE model_usage ADD COLUMN budget_scope TEXT NOT NULL DEFAULT 'general'"
            )
        connection.commit()
        return connection

    @contextmanager
    def _database(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _monthly_cost(connection: sqlite3.Connection, month_key: str) -> Decimal:
    rows = connection.execute(
        """SELECT status, reserved_cost_credits, actual_cost_credits
            FROM model_usage WHERE month_key=?""",
        (month_key,),
    ).fetchall()
    return sum(
        (Decimal(row[2]) if row[0] == "success" else Decimal(row[1]) for row in rows),
        start=Decimal("0"),
    )


def _token_usage(
    connection: sqlite3.Connection,
    *,
    budget_scope: str | None = None,
) -> TokenBudgetSnapshot:
    query = "SELECT status, prompt_tokens, completion_tokens, input_token_upper_bound, max_output_tokens FROM model_usage"
    parameters: tuple[str, ...] = ()
    if budget_scope in _P5_LOCALIZATION_BUDGET_SCOPES:
        placeholders = ",".join("?" for _ in _P5_LOCALIZATION_BUDGET_SCOPES)
        query += f" WHERE budget_scope IN ({placeholders})"
        parameters = _P5_LOCALIZATION_BUDGET_SCOPES
    elif budget_scope is not None:
        query += " WHERE budget_scope=?"
        parameters = (budget_scope,)
    rows = connection.execute(query, parameters).fetchall()
    confirmed = sum(
        (row[1] or 0) + (row[2] or 0)
        for row in rows
        if row[0] == "success"
    )
    reserved = sum(
        (row[3] or 0) + (row[4] or 0)
        for row in rows
        if row[0] in {"uncertain", "reserved"}
    )
    return TokenBudgetSnapshot(
        used_tokens=confirmed + reserved,
        confirmed_tokens=confirmed,
        reserved_tokens=reserved,
        request_count=len(rows),
        uncertain_requests=sum(row[0] == "uncertain" for row in rows),
        reserved_requests=sum(row[0] == "reserved" for row in rows),
    )
