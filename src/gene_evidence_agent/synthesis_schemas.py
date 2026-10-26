"""Independent, source-linked types for cross-publication research synthesis."""

from typing import Literal
from uuid import UUID

from pydantic import Field, HttpUrl, model_validator

from gene_evidence_agent.chat_schemas import ChatCitation, ChatUsage
from gene_evidence_agent.schemas import StrictModel


class SynthesisRequest(StrictModel):
    result_id: UUID


class Paper(StrictModel):
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{20}$")
    record_index: int = Field(ge=0)
    pmid: str = Field(pattern=r"^\d+$")
    doi: str | None = None
    title: str
    journal: str | None = None
    publication_date: str | None = None
    abstract_en: str = Field(min_length=1, max_length=10000)
    publication_types: list[str] = Field(default_factory=list,max_length=32)
    url: HttpUrl

    @model_validator(mode="after")
    def safe_source(self):
        ChatCitation(source_id=self.evidence_id, kind="evidence", title=self.title, url=self.url)
        return self


class Exclusion(StrictModel):
    record_index: int
    reason: Literal[
        "no_abstract", "duplicate", "invalid_record", "paper_limit", "input_budget",
        "language_not_reported", "non_english_abstract",
    ]


class StudyFact(StrictModel):
    text_zh: str | None = Field(default=None, min_length=1, max_length=400)
    quote_en: str | None = Field(default=None, min_length=1, max_length=1600)
    quote_origin: Literal['abstract','pubmed_publication_type']='abstract'

    @model_validator(mode="after")
    def require_pair(self):
        if (self.text_zh is None) != (self.quote_en is None):
            raise ValueError("a reported fact requires its English source quote")
        return self


CONTEXT_FIELDS = ("design", "population", "species", "cancer_subtype", "gene_variant", "cohort", "endpoint")


class StudyCard(StrictModel):
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{20}$")
    design: StudyFact = Field(default_factory=StudyFact)
    population: StudyFact = Field(default_factory=StudyFact)
    species: StudyFact = Field(default_factory=StudyFact)
    cancer_subtype: StudyFact = Field(default_factory=StudyFact)
    gene_variant: StudyFact = Field(default_factory=StudyFact)
    cohort: StudyFact = Field(default_factory=StudyFact)
    endpoint: StudyFact = Field(default_factory=StudyFact)
    findings: list[StudyFact] = Field(min_length=1, max_length=3)
    limitations: list[StudyFact] = Field(default_factory=list, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def normalize_unreported_context(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for name in CONTEXT_FIELDS:
                if data.get(name) is None:
                    data[name] = {}
        return data

    @model_validator(mode="after")
    def reported_findings(self):
        if any(fact.text_zh is None for fact in self.findings + self.limitations):
            raise ValueError("findings and limitations must include reported facts")
        return self


class Extraction(StrictModel):
    studies: list[StudyCard] = Field(min_length=1, max_length=10)


class SynthesisClaim(StrictModel):
    kind: Literal["common_finding", "difference", "conflict", "inference", "hypothesis"]
    text_zh: str = Field(min_length=1, max_length=700)
    basis_zh: str = Field(min_length=1, max_length=1000)
    basis_origin: Literal["legacy_model", "server_fact_catalog"] = "legacy_model"
    fact_refs: list[str] = Field(min_length=2, max_length=20)
    verification_direction_zh: str | None = Field(default=None, max_length=700)
    hypothesis_status: Literal["not_validated"] | None = None

    @model_validator(mode="after")
    def validate_hypothesis(self):
        if len(self.fact_refs) != len(set(self.fact_refs)):
            raise ValueError("fact references must be distinct")
        if self.kind == "hypothesis":
            if self.hypothesis_status != "not_validated" or not self.verification_direction_zh:
                raise ValueError("hypothesis requires unvalidated label and verification direction")
        elif self.hypothesis_status is not None or self.verification_direction_zh is not None:
            raise ValueError("hypothesis fields only belong to hypotheses")
        return self


class SynthesisAnalysis(StrictModel):
    claims: list[SynthesisClaim] = Field(default_factory=list, max_length=12)
    evidence_gaps: list[str] = Field(min_length=1, max_length=12)


class SynthesisReport(StrictModel):
    synthesis_id: UUID
    result_id: UUID
    status: Literal["completed", "partial", "failed", "insufficient_evidence", "mock"]
    candidate_count: int = Field(ge=0)
    papers: list[Paper] = Field(default_factory=list, max_length=10)
    excluded: list[Exclusion] = Field(default_factory=list)
    studies: list[StudyCard] = Field(default_factory=list, max_length=10)
    analysis: SynthesisAnalysis | None = None
    fact_catalog: dict[str, dict] = Field(default_factory=dict)
    independence_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejections: list[dict[str, str]] = Field(default_factory=list)
    source_outcomes: list[dict] = Field(default_factory=list)
    usage: list[ChatUsage] = Field(default_factory=list, max_length=2)
    error_code: str | None = Field(default=None, pattern=r"^[a-z_]{1,64}$")
    prompt_version: str = "cross-literature-v23-subtype-coverage-20261012"
    evidence_gaps_origin: Literal["legacy_model", "server_study_metadata"] = "legacy_model"
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    chinese_review: Literal["user_accepted"] = "user_accepted"

    @model_validator(mode="after")
    def complete_requires_analysis(self):
        if self.status == "completed" and (self.analysis is None or len(self.studies) < 2 or self.error_code):
            raise ValueError("completed synthesis requires at least two studies and a validated analysis")
        known = {paper.evidence_id for paper in self.papers}
        ids = [card.evidence_id for card in self.studies]
        if len(ids) != len(set(ids)) or not set(ids).issubset(known):
            raise ValueError("study identifiers must refer to selected papers")
        return self


class SynthesisJob(StrictModel):
    job_id: UUID
    result_id: UUID
    status: Literal["queued", "extracting", "synthesizing", "completed", "partial", "failed", "insufficient_evidence", "mock"]
    report: SynthesisReport | None = None
    error_code: str | None = Field(default=None, pattern=r"^[a-z_]{1,64}$")
