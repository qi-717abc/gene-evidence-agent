"""Typed contracts for the confirmation-gated guided research workflow."""

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictBool, field_validator

from gene_evidence_agent.schemas import StrictModel


class WorkflowSessionRequest(StrictModel):
    pass


class WorkflowMessageRequest(StrictModel):
    session_id: UUID
    message_id: UUID
    text: str = Field(min_length=1, max_length=3000)
    sources: list[Literal["pubmed", "uniprot", "opentargets"]] | None = Field(default=None, max_length=3)
    include_drugs: StrictBool | None = None
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=4)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise ValueError("message must contain valid nonblank text")
        return value.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or len(value) != len(set(value))):
            raise ValueError("sources must be non-empty and unique")
        return value


class WorkflowPrepareActionRequest(StrictModel):
    session_id: UUID
    action: Literal["localize", "synthesize"]


class WorkflowConfirmActionRequest(StrictModel):
    session_id: UUID
    action_id: UUID
    decision: Literal["confirm", "cancel"]
    sources: list[Literal["pubmed", "uniprot", "opentargets"]] | None = Field(default=None, max_length=3)
    include_drugs: StrictBool | None = None
    gene: str | None = Field(default=None, min_length=1, max_length=120)
    disease: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or len(value) != len(set(value))):
            raise ValueError("sources must be non-empty and unique")
        return value


class WorkflowSelectResultRequest(StrictModel):
    session_id: UUID
    result_id: UUID


class WorkflowBindSynthesisRequest(StrictModel):
    session_id: UUID
    result_id: UUID
    synthesis_id: UUID


class WorkflowResolveEntityRequest(StrictModel):
    session_id: UUID
    result_id: UUID
    decision: Literal["keep_ambiguous", "select_candidate"]
    entity_id: str | None = Field(default=None, min_length=1, max_length=160)
