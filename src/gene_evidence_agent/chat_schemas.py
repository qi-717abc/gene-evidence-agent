"""Server-owned contracts for research conversations, separate from translation."""

from typing import Literal
from uuid import UUID

from pydantic import Field, HttpUrl, StrictBool, field_validator, model_validator

from gene_evidence_agent.schemas import StrictModel


class ChatMessageRequest(StrictModel):
    session_id: UUID
    message_id: UUID
    text: str = Field(min_length=1, max_length=3000)
    result_id: UUID | None = None
    synthesis_id: UUID | None = None
    use_evidence: StrictBool = False
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=4)
    knowledge_ids: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 and c not in "\n\t" for c in value):
            raise ValueError("message must contain valid nonblank text")
        return value

    @model_validator(mode="after")
    def require_report_id(self):
        if self.synthesis_id is not None and self.knowledge_ids:
            raise ValueError("bound synthesis cannot import general knowledge cards")
        if len(self.knowledge_ids) != len(set(self.knowledge_ids)):
            raise ValueError("knowledge IDs must be distinct")
        if self.use_evidence and self.result_id is None:
            raise ValueError("evidence context requires a server report id")
        if self.synthesis_id is not None and (not self.use_evidence or self.result_id is None):
            raise ValueError("synthesis requires its original report context")
        return self


class ChatSessionRequest(StrictModel):
    pass


class ChatClearRequest(StrictModel):
    session_id: UUID


class ChatCitation(StrictModel):
    source_id: str = Field(min_length=1, max_length=160)
    kind: Literal["knowledge", "evidence"]
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    record_id: str | None = Field(default=None, max_length=200)
    record_index: int | None = Field(default=None, ge=0)

    @field_validator("url")
    @classmethod
    def require_safe_link(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https" or value.username or value.password:
            raise ValueError("citation links must be HTTPS without credentials")
        return value


class ChatUsage(StrictModel):
    status: Literal["reported", "unknown", "not_called"]
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_reported_usage(self):
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if self.status == "reported":
            if any(v is None for v in values):
                raise ValueError("reported usage requires all counts")
            if self.total_tokens != self.prompt_tokens + self.completion_tokens:
                raise ValueError("usage counts must reconcile")
        elif any(v is not None for v in values):
            raise ValueError("unreported usage cannot present counts as known")
        return self


ChatErrorCode = Literal[
    "model_disabled", "model_budget", "timeout", "rate_limited", "empty_content",
    "output_truncated", "invalid_model_output", "invalid_citation", "transport_error",
    "unknown_usage", "model_error", "context_too_large",
    "input_token_preflight_mismatch", "output_token_limit_exceeded",
    "unsupported_claim",
]


class ModelOutputDiagnostic(StrictModel):
    path: str = Field(min_length=1,max_length=80)
    error_type: str = Field(pattern=r'^[a-z_]{1,60}$')


class ChatReply(StrictModel):
    status: Literal["completed", "mock", "failed"]
    answer: str = Field(default="", max_length=40000)
    citations: list[ChatCitation] = Field(default_factory=list, max_length=20)
    knowledge_version: str = Field(min_length=1, max_length=64)
    knowledge_matched: bool
    result_id: UUID | None = None
    synthesis_id: UUID | None = None
    context_kind: Literal["knowledge", "evidence", "knowledge_and_evidence", "general", "mock", "synthesis"]
    usage: ChatUsage
    error_code: ChatErrorCode | None = None
    citation_diagnostic: Literal["unknown_id", "malformed_marker", "markers_mismatch", "missing_citation", "unexpected_url"] | None = None
    semantic_diagnostic: Literal['unsupported_study_design', 'unsupported_absence_claim',
                                 'source_terminology_mismatch', 'source_causality_mismatch'] | None = None
    semantic_rule: Literal["review", "in_vitro", "in_vivo", "clinical_trial",
                           "unperformed_assessment", "validation_absence", "category_absence"] | None = None
    semantic_context: Literal["report_meta","prospective","study_assertion"] | None = None
    answer_generation_method: Literal[
        "model_validated", "report_extractive_fallback", "report_fact_extractive", "report_review_checklist"
    ] = "model_validated"
    fallback_error_code: Literal["unsupported_claim"] | None = None
    model_output_diagnostics: list[ModelOutputDiagnostic] = Field(default_factory=list,max_length=12)

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.answer_generation_method == "report_review_checklist":
            if (self.status != "completed" or self.context_kind != "synthesis"
                    or self.synthesis_id is None or self.usage.status != "not_called"):
                raise ValueError("review checklist requires a bound report and no model call")
        if self.answer_generation_method == "report_extractive_fallback":
            if (self.status != "completed" or self.context_kind != "synthesis"
                    or self.synthesis_id is None or self.fallback_error_code != "unsupported_claim"):
                raise ValueError("report fallback requires completed bound-report output and its rejected-model reason")
        elif self.fallback_error_code is not None:
            raise ValueError("fallback reason requires a report fallback")
        if self.status == "failed":
            if self.error_code is None or self.answer or self.citations:
                raise ValueError("failure requires a code and no accepted answer")
        elif self.error_code is not None or not self.answer.strip():
            raise ValueError("completed answer requires text and no failure code")
        if self.status == "mock" and self.usage.status != "not_called":
            raise ValueError("mock replies cannot claim model usage")
        return self


class ChatJob(StrictModel):
    job_id: UUID
    session_id: UUID
    message_id: UUID
    status: Literal["queued", "running", "completed", "failed"]
    reply: ChatReply | None = None

    @model_validator(mode="after")
    def require_terminal_reply(self):
        if (self.status in {"completed", "failed"}) != (self.reply is not None):
            raise ValueError("only terminal jobs carry a reply")
        if self.reply is not None and (self.status == "failed") != (self.reply.status == "failed"):
            raise ValueError("job and reply statuses must agree")
        return self
