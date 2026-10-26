"""Structured, non-executing interpretation of a gene-disease evidence request."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from gene_evidence_agent.errors import InvalidInputError, ModelOutputError, ToolError, UpstreamError
from gene_evidence_agent.model_adapter import ModelAdapter, ModelMessage
from gene_evidence_agent.schemas import StrictModel

_MAX_QUERY_CHARS = 5000
_MVP_SOURCES = {"pubmed", "uniprot", "opentargets"}

_PLAN_PROMPT = """You are a narrow request planner for a gene-disease evidence retrieval system.
Return exactly one JSON object with these keys only:
intent, gene_query, disease_query, sources, clarification_question.
intent must be gene_disease_evidence, needs_clarification, or out_of_scope.
For gene_disease_evidence, sources must contain exactly pubmed, uniprot, and opentargets.
Use JSON null for every absent optional value; never use an empty string. For
gene_disease_evidence, clarification_question must be null. For other intents,
set unused gene_query or disease_query values to null.
For needs_clarification or out_of_scope, sources must be empty.
Do not invent a gene or disease absent from the user's request. Do not answer the
biomedical question, diagnose a person, guide individual patient management, make
personal treatment recommendations, create URLs, or provide tool arguments.
Classify personalized diagnosis/treatment requests as out_of_scope. If either
required entity is unclear or absent, request clarification.
Use an empty sources array unless intent is gene_disease_evidence. JSON only; no
Markdown fences or additional keys."""


class EvidencePlan(StrictModel):
    """Validated planner result. This object is descriptive and never executes tools."""

    intent: Literal["gene_disease_evidence", "needs_clarification", "out_of_scope"]
    gene_query: str | None = Field(default=None, max_length=300)
    disease_query: str | None = Field(default=None, max_length=300)
    sources: list[Literal["pubmed", "uniprot", "opentargets"]] = Field(default_factory=list, max_length=3)
    clarification_question: str | None = Field(default=None, max_length=1000)

    @field_validator("sources")
    @classmethod
    def require_unique_sources(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("sources must not contain duplicates")
        return value

    @field_validator("gene_query", "disease_query", "clarification_question")
    @classmethod
    def strip_or_reject_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("optional planner strings must be absent or non-empty")
        return cleaned

    @model_validator(mode="after")
    def validate_intent_fields(self) -> EvidencePlan:
        if self.intent == "gene_disease_evidence":
            if not self.gene_query or not self.disease_query or not self.sources:
                raise ValueError("evidence plans require a gene, disease, and at least one source")
            if set(self.sources) != _MVP_SOURCES:
                raise ValueError("evidence plans must use the fixed MVP source set")
            if self.clarification_question is not None:
                raise ValueError("evidence plans must not include a clarification question")
        elif self.intent == "needs_clarification":
            if not self.clarification_question:
                raise ValueError("clarification plans require a clarification question")
            if self.sources:
                raise ValueError("clarification plans must not request data sources")
        else:
            if self.sources:
                raise ValueError("out-of-scope plans must not request data sources")
        return self


class EvidencePlanner:
    """Ask the configured model for a small typed plan; never invokes evidence tools."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    async def plan(self, query: str) -> EvidencePlan:
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARS:
            raise InvalidInputError(
                f"Planner query must contain 1 to {_MAX_QUERY_CHARS} characters", source="planner"
            )
        try:
            response = await self._adapter.complete(
                [
                    ModelMessage(role="system", content=_PLAN_PROMPT),
                    ModelMessage(role="user", content=query.strip()),
                ],
                max_output_tokens=300,
            )
        except InvalidInputError:
            raise
        except ToolError:
            raise
        except Exception as error:
            raise UpstreamError("Planner API call failed", source="planner") from error

        if response.finish_reason is not None and response.finish_reason.casefold() in {
            "length",
            "max_tokens",
        }:
            raise ModelOutputError(
                "Planner output reached the model token limit",
                failure_code="output_truncated",
                source="planner",
            )
        try:
            payload = json.loads(response.text)
        except (ValueError, TypeError) as error:
            raise ModelOutputError(
                "Planner returned invalid JSON",
                failure_code="invalid_json",
                source="planner",
            ) from error
        try:
            return EvidencePlan.model_validate(payload)
        except (ValidationError, TypeError, ValueError) as error:
            raise ModelOutputError(
                "Planner JSON did not match the required schema",
                failure_code="invalid_schema",
                source="planner",
            ) from error
