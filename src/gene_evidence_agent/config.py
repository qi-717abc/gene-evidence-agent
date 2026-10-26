"""Typed application settings loaded from environment variables or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for API access and bounded external requests.

    Model credentials are optional while the deterministic data-source pipeline
    is developed. A caller that invokes a model must validate its credentials
    before making the external request.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_provider: Literal["openai_compatible"] = "openai_compatible"
    model_name: str | None = None
    model_base_url: AnyHttpUrl | None = None
    model_api_key: SecretStr | None = None
    model_enabled: bool = False
    # Optional provider-specific parameter scoped to Chinese localization only.
    chinese_thinking_mode: Literal["enabled", "disabled"] | None = None
    chinese_claims_mode: Literal["model", "extractive"] = "model"
    # Extractive mode separates sources; model-claim mode keeps the legacy
    # cross-source batch contract. 10 explicitly selects legacy batch reading.
    chinese_translation_batch_size: int = Field(default=1,ge=1,le=10)
    model_monthly_budget_credits: Decimal = Field(default=Decimal("90.00"), gt=0)
    model_input_price_credits_per_million_tokens: Decimal = Field(default=Decimal("0.8"), ge=0)
    model_output_price_credits_per_million_tokens: Decimal = Field(default=Decimal("2.7"), ge=0)
    model_context_window_tokens: int = Field(default=1_000_000, ge=1, le=1_000_000)
    # User-authorized policy: keep uncertain reserves but do not freeze on count.
    # Set true to restore the baseline gate without changing the ledger.
    model_freeze_on_uncertain_usage: bool = False
    model_allowed_uncertain_requests: int = Field(default=0, ge=0)
    model_max_output_tokens: int = Field(default=200000, ge=1, le=1_000_000)
    # Model completions can take longer than public biomedical API lookups.
    model_request_timeout_seconds: float = Field(default=600.0, gt=0, le=1800)
    model_usage_db_path: Path = Path(".gene-evidence-agent/model_usage.sqlite3")

    agent_run_timeout_seconds: float = Field(default=900.0, gt=0, le=3600)
    ncbi_email: str | None = None
    ncbi_api_key: SecretStr | None = None

    request_timeout_seconds: float = Field(default=60.0, gt=0, le=120)
    max_results_per_source: int = Field(default=10, ge=1, le=50)
    cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("model_name", "ncbi_email", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("ncbi_email")
    @classmethod
    def validate_ncbi_email_shape(cls, value: str | None) -> str | None:
        if value is not None and ("@" not in value or value.startswith("@") or value.endswith("@")):
            raise ValueError("GEA_NCBI_EMAIL must be a valid email address")
        if value is not None and value.rsplit("@", 1)[-1].casefold() in {
            "example.com",
            "example.net",
            "example.org",
        }:
            return None
        return value

    @field_validator("ncbi_api_key", mode="before")
    @classmethod
    def ignore_ncbi_api_key_placeholders(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw_value, str) and raw_value.strip().casefold() in {
            "replace-with-your-own-secret",
            "your-ncbi-api-key",
            "your_ncbi_api_key",
        }:
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()

