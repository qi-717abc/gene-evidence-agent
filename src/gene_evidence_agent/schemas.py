"""Validated biomedical entities and source-grounded evidence models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class StrictModel(BaseModel):
    """Base class that rejects undeclared fields and post-creation mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ToolStatus(StrEnum):
    """Stable outcome categories shared by every evidence-source adapter."""

    SUCCESS = "success"
    EMPTY = "empty"
    INVALID_INPUT = "invalid_input"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UPSTREAM_ERROR = "upstream_error"


class BiomedicalEntity(StrictModel):
    """An input or discovered biomedical entity with optional normalized ID."""

    id: str = Field(min_length=1, max_length=160)
    entity_type: Literal["gene", "disease", "drug", "variant", "protein"]
    query_name: str = Field(min_length=1, max_length=300)
    canonical_name: str | None = Field(default=None, max_length=300)
    identifier: str | None = Field(default=None, max_length=160)
    namespace: str | None = Field(default=None, max_length=80)
    organism_tax_id: int | None = Field(default=None, ge=1)
    resolution_status: Literal["unresolved", "resolved", "ambiguous"] = "unresolved"
    resolution_score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("id", "query_name", "canonical_name", "identifier", "namespace")
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("string fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> BiomedicalEntity:
        if (self.identifier is None) != (self.namespace is None):
            raise ValueError("identifier and namespace must be provided together")
        if self.resolution_status == "resolved" and self.identifier is None:
            raise ValueError("resolved entities must have an identifier and namespace")
        return self


class SourceReference(StrictModel):
    """Provenance for a single source record returned by an upstream database."""

    source_database: str = Field(min_length=1, max_length=100)
    record_id: str = Field(min_length=1, max_length=300)
    record_url: HttpUrl
    retrieved_at: datetime

    @field_validator("source_database", "record_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("string fields must not be blank")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value


class EvidenceRecord(StrictModel):
    """A claim or observation linked to one or more report entities and a source."""

    evidence_type: Literal[
        "publication",
        "protein_annotation",
        "target_disease_association",
        "disease_drug_candidate",
        "variant_interpretation",
        "clinical_trial",
        "other",
    ]
    source: SourceReference
    also_reported_by: list[SourceReference] = Field(default_factory=list, max_length=20)
    entity_ids: list[str] = Field(min_length=1)
    title: str | None = Field(default=None, max_length=1000)
    summary: str = Field(min_length=1, max_length=10000)
    journal: str | None = Field(default=None, max_length=500)
    publication_date: str | None = Field(default=None, max_length=300)
    doi: str | None = Field(default=None, max_length=300)
    pmid: str | None = Field(default=None, pattern=r"^\d+$", max_length=20)
    publication_types: list[str] = Field(default_factory=list,max_length=32)
    publication_languages: list[str] = Field(default_factory=list, max_length=8)
    score: float | None = Field(default=None, ge=0, le=1)
    raw_payload_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    warnings: list[str] = Field(default_factory=list)

    @field_validator('publication_types')
    @classmethod
    def bounded_publication_types(cls,values):
        if any(not value.strip() or len(value)>160 for value in values) or len(values)!=len(set(values)):
            raise ValueError('publication types must be distinct bounded source strings')
        return values

    @field_validator("publication_languages")
    @classmethod
    def bounded_publication_languages(cls, values):
        normalized = [value.strip().casefold() for value in values]
        if any(not value or len(value) > 40 for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("publication languages must be distinct bounded source strings")
        return normalized

    @field_validator("entity_ids")
    @classmethod
    def require_distinct_entity_ids(cls, value: list[str]) -> list[str]:
        cleaned = [entity_id.strip() for entity_id in value]
        if any(not entity_id for entity_id in cleaned):
            raise ValueError("entity_ids must not contain blank values")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entity_ids must be unique within an evidence record")
        return cleaned

    @field_validator("summary")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        return value

    @field_validator("title", "journal", "publication_date", "doi", "pmid")
    @classmethod
    def reject_blank_optional_metadata(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional metadata must be absent or non-blank")
        return value


class ToolResult(StrictModel):
    """Normalized response envelope for a single upstream source query."""

    status: ToolStatus
    source: str = Field(min_length=1, max_length=100)
    query: dict[str, Any] = Field(default_factory=dict)
    records: list[EvidenceRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    request_id: UUID

    @field_validator("source")
    @classmethod
    def reject_blank_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source must not be blank")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_status_and_records(self) -> ToolResult:
        if self.status == ToolStatus.SUCCESS and not self.records:
            raise ValueError("successful tool results must contain at least one record")
        if self.status != ToolStatus.SUCCESS and self.records:
            raise ValueError("only successful tool results may contain records")
        return self


class SourceOutcome(StrictModel):
    """Sanitized per-source status for a multi-source collection run."""

    source: Literal["pubmed", "uniprot", "opentargets"]
    status: Literal[
        "not_requested",
        "success",
        "empty",
        "ambiguous",
        "partial_success",
        "invalid_input",
        "rate_limited",
        "timeout",
        "upstream_error",
    ]
    record_count: int = Field(default=0, ge=0)
    error_code: Literal["invalid_input", "rate_limited", "timeout", "upstream_error"] | None = None


class EvidenceReport(StrictModel):
    """Source-grounded result returned for a user query."""

    query: str = Field(min_length=1, max_length=5000)
    entities: list[BiomedicalEntity] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    source_outcomes: list[SourceOutcome] = Field(default_factory=list, max_length=3)
    summary: str | None = Field(default=None, max_length=20000)
    conflicting_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, max_length=100)
    evidence_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_evidence_entity_references(self) -> EvidenceReport:
        known_entity_ids = {entity.id for entity in self.entities}
        if len(known_entity_ids) != len(self.entities):
            raise ValueError("entity ids must be unique within a report")
        unknown_ids = {
            entity_id
            for record in self.evidence
            for entity_id in record.entity_ids
            if entity_id not in known_entity_ids
        }
        if unknown_ids:
            raise ValueError(f"evidence refers to entities absent from report: {sorted(unknown_ids)}")
        source_names = [outcome.source for outcome in self.source_outcomes]
        if len(source_names) != len(set(source_names)):
            raise ValueError("source outcome names must be unique within a report")
        return self

