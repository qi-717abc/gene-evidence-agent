"""Finite-state orchestration for source-grounded gene-disease evidence queries."""

from __future__ import annotations

import asyncio
import hashlib
import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from gene_evidence_agent.chinese_report import (
    AGENT_RUN_STATUS_LABELS_ZH,
    ChineseEvidenceReport,
    agent_run_warning_label_zh,
    localize_report,
    validate_localized_report_against,
)
from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import InvalidInputError, ModelCallError, RateLimitedError, ToolError
from gene_evidence_agent.model_adapter import ModelAdapter
from gene_evidence_agent.pipeline import GeneEvidencePipeline
from gene_evidence_agent.planner import EvidencePlan, EvidencePlanner
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.schemas import BiomedicalEntity, EvidenceReport, StrictModel
from gene_evidence_agent.summarizer import EvidenceSummarizer, EvidenceSummary

MAX_AGENT_ROUNDS = 3
_SAFETY_NOTICE = (
    "This agent retrieves research evidence; it does not provide a diagnosis or personal treatment advice. "
    "Clinical decisions require a qualified healthcare professional."
)


class AgentPhase(StrEnum):
    """The only permitted stages; each stage may be entered at most once."""

    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    SUMMARIZATION = "summarization"


class AgentRunResult(StrictModel):
    """A bounded run result with explicit terminal status and phase trace."""

    status: Literal[
        "completed",
        "partial_success",
        "needs_clarification",
        "out_of_scope",
        "upstream_error",
    ]
    plan: EvidencePlan | None = None
    report: EvidenceReport | None = None
    summary: EvidenceSummary | None = None
    chinese_report: ChineseEvidenceReport | None = None
    status_label_zh: str | None = None
    phases: list[AgentPhase] = Field(min_length=1, max_length=MAX_AGENT_ROUNDS)
    rounds_used: int = Field(ge=1, le=MAX_AGENT_ROUNDS)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    warnings_zh: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def add_labels_for_chinese_runs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if payload.get("chinese_report") is not None:
            status = payload.get("status")
            if payload.get("status_label_zh") is None and isinstance(status, str):
                payload["status_label_zh"] = AGENT_RUN_STATUS_LABELS_ZH.get(status, "运行状态未知")
            warnings = payload.get("warnings", [])
            if payload.get("warnings_zh") is None and isinstance(warnings, list) and all(
                isinstance(item, str) for item in warnings
            ):
                payload["warnings_zh"] = [agent_run_warning_label_zh(item) for item in warnings]
        return payload

    @model_validator(mode="after")
    def validate_trace_and_terminal_state(self) -> AgentRunResult:
        if len(self.phases) != self.rounds_used:
            raise ValueError("rounds_used must equal the number of executed phases")
        if len(self.phases) != len(set(self.phases)):
            raise ValueError("an agent phase may execute only once")
        if self.phases[0] != AgentPhase.PLANNING:
            raise ValueError("every agent run must begin with planning")
        if AgentPhase.RETRIEVAL in self.phases and self.phases.index(AgentPhase.RETRIEVAL) != 1:
            raise ValueError("retrieval must immediately follow planning")
        if AgentPhase.SUMMARIZATION in self.phases:
            if self.phases[-1] != AgentPhase.SUMMARIZATION or AgentPhase.RETRIEVAL not in self.phases:
                raise ValueError("summarization must be the final stage after retrieval")
        if self.status in {"completed", "partial_success"} and (self.plan is None or self.report is None):
            raise ValueError("successful runs require a validated plan and evidence report")
        if self.status in {"needs_clarification", "out_of_scope"}:
            if self.plan is None or self.report is not None or self.rounds_used != 1:
                raise ValueError("clarification and out-of-scope runs must stop after planning")
        if self.status == "upstream_error" and self.report is not None:
            raise ValueError("upstream-error runs must not present a completed evidence report")
        if self.chinese_report is not None:
            if self.status_label_zh != AGENT_RUN_STATUS_LABELS_ZH[self.status]:
                raise ValueError("Chinese Agent status label does not match the raw status")
            if self.warnings_zh != [agent_run_warning_label_zh(item) for item in self.warnings]:
                raise ValueError("Chinese Agent warning labels do not match the raw warning codes")
            if self.report is None:
                raise ValueError("a Chinese view requires the unchanged original evidence report")
            validate_localized_report_against(self.report, self.chinese_report)
        elif self.status_label_zh is not None or self.warnings_zh is not None:
            raise ValueError("Chinese Agent labels require a Chinese report")
        return self


class GeneEvidenceAgent:
    """Run at most planning, deterministic retrieval, and evidence summarization."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        settings: Settings | None = None,
        pipeline: GeneEvidencePipeline | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._adapter = adapter
        self._planner = EvidencePlanner(adapter)
        self._pipeline = pipeline or GeneEvidencePipeline(self._settings)
        self._summarizer = EvidenceSummarizer(adapter)

    async def run(self, query: str, *, language: Literal["en", "zh"] = "en") -> AgentRunResult:
        """Execute the fixed plan -> retrieve -> summarize/localize workflow once."""

        if language not in {"en", "zh"}:
            raise InvalidInputError("language must be 'en' or 'zh'", source="agent")
        if isinstance(query, str):
            ensure_no_sensitive_query(query)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.agent_run_timeout_seconds
        phases = [AgentPhase.PLANNING]
        try:
            plan = await asyncio.wait_for(
                self._planner.plan(query), timeout=max(0.0, deadline - loop.time())
            )
        except TimeoutError:
            return AgentRunResult(
                status="upstream_error",
                phases=phases,
                rounds_used=len(phases),
                warnings=["planner_timeout"],
            )
        except InvalidInputError:
            raise
        except ToolError as error:
            return AgentRunResult(
                status="upstream_error",
                phases=phases,
                rounds_used=len(phases),
                warnings=[_failure_warning("planner", error)],
            )

        if plan.intent == "needs_clarification":
            return AgentRunResult(
                status="needs_clarification",
                plan=plan,
                phases=phases,
                rounds_used=len(phases),
            )
        if plan.intent == "out_of_scope":
            return AgentRunResult(
                status="out_of_scope",
                plan=plan,
                phases=phases,
                rounds_used=len(phases),
            )

        # EvidencePlan validates this exact source set; the pipeline has no model-controlled tools.
        assert plan.gene_query is not None and plan.disease_query is not None
        phases.append(AgentPhase.RETRIEVAL)
        gene = _query_entity("gene", plan.gene_query)
        disease = _query_entity("disease", plan.disease_query)
        try:
            collection = await asyncio.wait_for(
                self._pipeline.collect(
                    gene, disease, limit=self._settings.max_results_per_source
                ),
                timeout=max(0.0, deadline - loop.time()),
            )
        except TimeoutError:
            return AgentRunResult(
                status="upstream_error",
                plan=plan,
                phases=phases,
                rounds_used=len(phases),
                warnings=["retrieval_timeout"],
            )
        except ToolError as error:
            return AgentRunResult(
                status="upstream_error",
                plan=plan,
                phases=phases,
                rounds_used=len(phases),
                warnings=[_failure_warning("retrieval", error)],
            )

        report = collection.to_evidence_report(query=query.strip())
        if _SAFETY_NOTICE not in report.limitations:
            report = report.model_copy(update={"limitations": [*report.limitations, _SAFETY_NOTICE]})
        phases.append(AgentPhase.SUMMARIZATION)
        if language == "zh":
            localized = await localize_report(
                report,
                settings=self._settings,
                adapter=self._adapter,
                timeout_seconds=max(0.0, deadline - loop.time()),
            )
            if localized.translation_status in {
                "failed_fallback",
                "partial_output_limit",
                "partial_validation",
            }:
                reason = localized.translation_error_code or localized.translation_status
                return AgentRunResult(
                    status="partial_success",
                    plan=plan,
                    report=report,
                    chinese_report=localized,
                    phases=phases,
                    rounds_used=len(phases),
                    warnings=[f"chinese_localization_{reason}"],
                )
            return AgentRunResult(
                status="completed",
                plan=plan,
                report=report,
                chinese_report=localized,
                phases=phases,
                rounds_used=len(phases),
            )
        try:
            summary = await asyncio.wait_for(
                self._summarizer.summarize(report), timeout=max(0.0, deadline - loop.time())
            )
        except TimeoutError:
            return AgentRunResult(
                status="partial_success",
                plan=plan,
                report=report,
                phases=phases,
                rounds_used=len(phases),
                warnings=["summarizer_timeout"],
            )
        except ToolError as error:
            # Keep deterministic database evidence usable when the optional LLM summary fails.
            return AgentRunResult(
                status="partial_success",
                plan=plan,
                report=report,
                phases=phases,
                rounds_used=len(phases),
                warnings=[_failure_warning("summarizer", error)],
            )

        return AgentRunResult(
            status="completed",
            plan=plan,
            report=report,
            summary=summary,
            phases=phases,
            rounds_used=len(phases),
        )


def _query_entity(entity_type: Literal["gene", "disease"], query_name: str) -> BiomedicalEntity:
    normalized = query_name.casefold().strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return BiomedicalEntity(
        id=f"query:{entity_type}:{digest}",
        entity_type=entity_type,
        query_name=query_name,
    )


def _failure_warning(stage: str, error: ToolError) -> str:
    if isinstance(error, ModelCallError):
        code = f"http_{error.status_code}" if error.status_code is not None else error.failure_code
        if error.finish_reason is not None:
            code = f"{code}_finish_reason_{error.finish_reason}"
        return f"{stage}_{code}"
    code = getattr(error.status, "value", "upstream_error")
    if code not in {"invalid_input", "rate_limited", "timeout", "upstream_error"}:
        code = "upstream_error"
    if isinstance(error, RateLimitedError):
        cooldown = (
            f"_retry_after_{math.ceil(error.retry_after_seconds)}s"
            if error.retry_after_seconds is not None
            else ""
        )
        if error.provider_code is not None:
            return f"{stage}_rate_limited_provider_{error.provider_code}{cooldown}"
        if cooldown:
            return f"{stage}_rate_limited{cooldown}"
    return f"{stage}_{code}"
