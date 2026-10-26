"""Bounded in-memory sessions and background jobs for the research sidebar."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import asyncio
import hashlib
import json
import logging
from threading import RLock
from time import monotonic
from typing import Callable
from uuid import UUID, uuid4

from pydantic import ValidationError

from gene_evidence_agent.chat_schemas import ChatJob, ChatMessageRequest, ChatReply, ChatUsage
from gene_evidence_agent.knowledge_store import KnowledgeStore
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.errors import InvalidInputError
from gene_evidence_agent.synthesis import SynthesisEngine
from gene_evidence_agent.synthesis_schemas import SynthesisJob
from gene_evidence_agent.staged_synthesis import (
    EnglishSynthesisEngine,
    SynthesisAction,
    SynthesisActionRequest,
    SynthesisPlan,
    SynthesisStageError,
    StagedSynthesisJob,
    StagedSynthesisReport,
    build_report,
)
from gene_evidence_agent.model_tokenizer import ModelTokenizerError
from gene_evidence_agent.attachments import AttachmentStore, AttachmentError

SESSION_TTL_SECONDS = 1800
MAX_SESSIONS = 100
MAX_REPORTS = 100
MAX_TURNS = 20
MAX_IN_FLIGHT = 4
logger = logging.getLogger(__name__)


class ChatServiceError(Exception):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code, self.status = code, status


@dataclass
class Session:
    touched: float
    history: list[tuple[str, str]] = field(default_factory=list)
    requests: dict[UUID, ChatMessageRequest] = field(default_factory=dict)
    jobs: dict[UUID, ChatJob] = field(default_factory=dict)
    busy: bool = False
    bound: bool = False
    result_id: UUID | None = None
    synthesis_id: UUID | None = None
    attachment_ids: list[UUID] = field(default_factory=list)


@dataclass
class StagedSynthesisRun:
    job: StagedSynthesisJob
    prepared: dict
    plan: SynthesisPlan | None = None
    plan_draft: object | None = None
    studies: list = field(default_factory=list)
    analysis: object | None = None
    catalog: dict = field(default_factory=dict)
    independence_notes: list[str] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    processed_actions: dict[UUID, str] = field(default_factory=dict)


class ResearchChatService:
    def __init__(self, *, mode="mock", responder: Callable | None = None,
                 knowledge: KnowledgeStore | None = None, clock=monotonic,
                 synthesis_engine=None, synthesis_responder=None, staged_synthesis_engine=None):
        self.mode = mode
        self.knowledge = knowledge or KnowledgeStore()
        self.responder = responder
        self.clock = clock
        self.sessions: dict[UUID, Session] = {}
        self.attachments = AttachmentStore(clock=clock)
        self.reports: dict[UUID, tuple[float, dict]] = {}
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT, thread_name_prefix="research-chat")
        self.active = 0
        self.synthesis_engine = synthesis_engine or SynthesisEngine(mode="mock" if mode == "mock" else "live")
        self.staged_synthesis_engine = staged_synthesis_engine or EnglishSynthesisEngine(
            mode="mock" if mode == "mock" else "live"
        )
        self.synthesis_responder = synthesis_responder
        self.synthesis_jobs = {}
        self.synthesis_reports = {}
        self.staged_synthesis_runs: dict[tuple[UUID, UUID], StagedSynthesisRun] = {}
        self.staged_synthesis_by_result: dict[UUID, UUID] = {}

    def close(self):
        # In-flight adapter requests continue to settle their persistent usage.
        self.executor.shutdown(wait=True, cancel_futures=False)

    def health(self):
        extra = self.responder.health() if self.responder is not None and hasattr(self.responder, "health") else {}
        synthesis_health = self.staged_synthesis_engine.health()
        followup_ready = self.mode == "mock" or (
            self.synthesis_responder is not None and
            (not hasattr(self.synthesis_responder, "health") or self.synthesis_responder.health()["enabled"])
        )
        return {"mode": self.mode, "enabled": self.mode == "mock" or self.responder is not None,
                "knowledge_version": self.knowledge.version,
                "knowledge_card_count": len(self.knowledge.cards), **extra,
                "synthesis_enabled": followup_ready,
                "synthesis": synthesis_health}

    def _purge(self):
        cutoff = self.clock() - SESSION_TTL_SECONDS
        self.sessions = {key: session for key, session in self.sessions.items()
                         if session.busy or session.touched >= cutoff}
        pinned = {key for key, job in self.synthesis_jobs.items() if job.status in {"queued", "extracting", "synthesizing"}}
        self.reports = {key: item for key, item in self.reports.items() if item[0] >= cutoff or key in pinned}
        self.synthesis_jobs = {key: job for key, job in self.synthesis_jobs.items() if key in self.reports}
        self.synthesis_reports = {key: value for key, value in self.synthesis_reports.items() if value.result_id in self.reports}
        active_staged = {
            (result_id, job_id)
            for (result_id, job_id), run in self.staged_synthesis_runs.items()
            if run.job.status in {"planning", "extracting", "synthesizing", "translating"}
        }
        self.staged_synthesis_runs = {
            key: run for key, run in self.staged_synthesis_runs.items()
            if key[0] in self.reports and (self.reports[key[0]][0] >= cutoff or key in active_staged)
        }
        self.staged_synthesis_by_result = {
            result_id: job_id for result_id, job_id in self.staged_synthesis_by_result.items()
            if (result_id, job_id) in self.staged_synthesis_runs
        }

    def create_session(self):
        with self.lock:
            self._purge()
            if len(self.sessions) >= MAX_SESSIONS:
                raise ChatServiceError("capacity_reached", 429)
            key = uuid4()
            self.sessions[key] = Session(self.clock())
            return {"session_id": str(key), "expires_in_seconds": SESSION_TTL_SECONDS}

    def register_report(self, report: dict) -> str:
        with self.lock:
            self._purge()
            if len(self.reports) >= MAX_REPORTS:
                eligible = [key for key in self.reports if key not in self.synthesis_jobs or self.synthesis_jobs[key].status not in {"queued", "extracting", "synthesizing"}]
                if not eligible:
                    raise ChatServiceError("capacity_reached", 429)
                oldest = min(eligible, key=lambda key: self.reports[key][0])
                del self.reports[oldest]
            key = uuid4()
            self.reports[key] = (self.clock(), deepcopy(report))
            return str(key)

    def start_staged_synthesis(self, result_id):
        """Preflight locally, then wait for an explicit plan-stage confirmation."""
        with self.lock:
            self._purge()
            if result_id not in self.reports:
                raise ChatServiceError("report_not_found", 404)
            prior_id = self.staged_synthesis_by_result.get(result_id)
            prior = self.staged_synthesis_runs.get((result_id, prior_id)) if prior_id else None
            if prior is not None:
                return prior.job
            if not self.staged_synthesis_engine.health()["enabled"]:
                raise ChatServiceError("synthesis_disabled", 503)
            source = deepcopy(self.reports[result_id][1])
            try:
                prepared = self.staged_synthesis_engine.preflight(result_id, source)
            except SynthesisStageError as error:
                raise ChatServiceError(str(error), 422) from None
            except Exception:
                raise ChatServiceError("synthesis_preflight_failed", 422) from None
            job_id = uuid4()
            if self.mode == "mock":
                report = build_report(result_id, prepared, status="mock")
                job = StagedSynthesisJob(
                    job_id=job_id, result_id=result_id, status="mock",
                    candidate_count=prepared["candidate_count"], papers=prepared["papers"],
                    excluded=prepared["excluded"], report=report,
                )
                self.staged_synthesis_runs[(result_id, job_id)] = StagedSynthesisRun(job, prepared)
                self.staged_synthesis_by_result[result_id] = job_id
                return job
            if len(prepared["papers"]) < 2:
                report = build_report(result_id, prepared, status="insufficient_evidence")
                job = StagedSynthesisJob(
                    job_id=job_id, result_id=result_id, status="insufficient_evidence",
                    candidate_count=prepared["candidate_count"], papers=prepared["papers"],
                    excluded=prepared["excluded"], report=report,
                )
                self.staged_synthesis_runs[(result_id, job_id)] = StagedSynthesisRun(job, prepared)
                self.staged_synthesis_by_result[result_id] = job_id
                return job
            action = SynthesisAction(action_id=uuid4(), kind="start_plan")
            job = StagedSynthesisJob(
                job_id=job_id, result_id=result_id, status="awaiting_plan_start",
                candidate_count=prepared["candidate_count"], papers=prepared["papers"],
                excluded=prepared["excluded"],
                selected_evidence_ids=[paper.evidence_id for paper in prepared["papers"]],
                current_action=action,
            )
            self.staged_synthesis_runs[(result_id, job_id)] = StagedSynthesisRun(job, prepared)
            self.staged_synthesis_by_result[result_id] = job_id
            return job

    def get_staged_synthesis_job(self, result_id, job_id):
        with self.lock:
            self._purge()
            run = self.staged_synthesis_runs.get((result_id, job_id))
            if run is None:
                raise ChatServiceError("job_not_found", 404)
            return run.job

    def confirm_staged_synthesis(self, request: SynthesisActionRequest):
        with self.lock:
            self._purge()
            key = (request.result_id, request.job_id)
            run = self.staged_synthesis_runs.get(key)
            if run is None:
                raise ChatServiceError("job_not_found", 404)
            fingerprint = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            previous = run.processed_actions.get(request.action_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ChatServiceError("action_id_conflict", 409)
                return run.job
            action = run.job.current_action
            if action is None or action.action_id != request.action_id:
                raise ChatServiceError("action_stale", 409)
            if self.staged_synthesis_by_result.get(request.result_id) != request.job_id:
                raise ChatServiceError("result_version_changed", 409)

            run.processed_actions[request.action_id] = fingerprint
            if request.decision == "cancel":
                if action.kind == "translation_decision":
                    if run.job.report is not None:
                        report = run.job.report.model_copy(update={"translation_status": "declined"})
                        run.job = run.job.model_copy(update={
                            "status": "completed", "report": report, "current_action": None,
                        })
                        self.synthesis_reports[report.synthesis_id] = report
                    else:
                        run.job = run.job.model_copy(update={"status": "cancelled", "current_action": None})
                else:
                    run.job = run.job.model_copy(update={"status": "cancelled", "current_action": None})
                return run.job

            if request.decision == "revise_plan":
                if action.kind != "approve_cards" or run.plan is None:
                    raise ChatServiceError("action_stale", 409)
                next_action = SynthesisAction(action_id=uuid4(), kind="approve_plan")
                run.studies = []
                run.analysis = None
                run.catalog = {}
                run.independence_notes = []
                run.job = run.job.model_copy(update={
                    "status": "awaiting_plan_review", "current_action": next_action,
                    "studies": [], "report": None,
                })
                return run.job

            expected = {
                "start_plan": ("awaiting_plan_start", "planning"),
                "approve_plan": ("awaiting_plan_review", "extracting"),
                "approve_cards": ("awaiting_study_review", "synthesizing"),
                "translation_decision": ("awaiting_translation_decision", "translating"),
            }
            if action.kind not in expected or run.job.status != expected[action.kind][0]:
                raise ChatServiceError("action_stale", 409)
            selected_ids = request.selected_evidence_ids or run.job.selected_evidence_ids
            plan_inputs_changed = False
            if action.kind in {"start_plan", "approve_plan"}:
                try:
                    selected_ids = [
                        paper.evidence_id
                        for paper in self.staged_synthesis_engine.paper_map(run.job.papers, selected_ids)
                    ]
                except SynthesisStageError as error:
                    run.processed_actions.pop(request.action_id, None)
                    raise ChatServiceError(str(error), 400) from None
                focus = request.focus_questions_zh or run.job.focus_questions_zh
                if not focus:
                    focus = ["比较研究对象、癌症亚型、基因背景、主要发现、研究差异和摘要层面的证据缺口。"]
                if len(focus) > 8 or any(not item.strip() or len(item) > 500 for item in focus):
                    run.processed_actions.pop(request.action_id, None)
                    raise ChatServiceError("invalid_focus_questions", 400)
                run.job = run.job.model_copy(update={
                    "selected_evidence_ids": list(selected_ids),
                    "focus_questions_zh": list(focus),
                })
                if action.kind == "approve_plan" and run.plan is not None:
                    plan_inputs_changed = (
                        list(selected_ids) != run.plan.selected_evidence_ids
                        or list(focus) != run.plan.focus_questions_zh
                    )
                    digest = hashlib.sha256(json.dumps({
                        "source": run.prepared["input_digest"],
                        "selected": list(selected_ids),
                        "focus": list(focus),
                        "dimensions": run.plan.comparison_dimensions_zh,
                        "questions": run.plan.comparison_questions_zh,
                    }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                    run.plan = run.plan.model_copy(update={
                        "plan_id": uuid4(), "selected_evidence_ids": list(selected_ids),
                        "focus_questions_zh": list(focus), "input_digest": digest,
                    })
                    run.job = run.job.model_copy(update={"plan": run.plan})
                    if plan_inputs_changed:
                        # The generated plan was based on the previous paper set/focus.
                        # Re-plan before asking for the plan approval that unlocks extraction.
                        run.plan = None
                        run.plan_draft = None
                        run.job = run.job.model_copy(update={"plan": None, "studies": [], "report": None})
            if self.active >= MAX_IN_FLIGHT:
                run.processed_actions.pop(request.action_id, None)
                raise ChatServiceError("capacity_reached", 429)
            stage = "planning" if plan_inputs_changed else expected[action.kind][1]
            run.job = run.job.model_copy(update={"status": stage, "current_action": None})
            self.active += 1
            self.executor.submit(self._run_staged_synthesis_stage, key, stage)
            return run.job

    def _run_staged_synthesis_stage(self, key, stage):
        result_id, job_id = key
        with self.lock:
            run = self.staged_synthesis_runs.get(key)
            if run is None:
                self.active = max(0, self.active - 1)
                return
            job = run.job
            source = deepcopy(self.reports.get(result_id, (0, {}))[1])
            selected_ids = list(job.selected_evidence_ids)
            papers = self.staged_synthesis_engine.paper_map(job.papers, selected_ids) if selected_ids else []
            report_prepared = {**run.prepared, "papers": papers}
            plan = run.plan
            draft = run.plan_draft
            studies = list(run.studies)
            analysis = run.analysis
            catalog = deepcopy(run.catalog)
            notes = list(run.independence_notes)
        try:
            if stage == "planning":
                draft, usage = asyncio.run(self.staged_synthesis_engine.plan(result_id, papers, job.focus_questions_zh))
                digest = hashlib.sha256(json.dumps({
                    "source": run.prepared["input_digest"],
                    "selected": selected_ids,
                    "focus": job.focus_questions_zh,
                    "dimensions": draft.comparison_dimensions_zh,
                    "questions": draft.comparison_questions_zh,
                }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                plan = SynthesisPlan(
                    plan_id=uuid4(), result_id=result_id, selected_evidence_ids=selected_ids,
                    focus_questions_zh=job.focus_questions_zh,
                    comparison_dimensions_zh=draft.comparison_dimensions_zh,
                    comparison_questions_zh=draft.comparison_questions_zh,
                    scope_limitations_zh=draft.scope_limitations_zh, input_digest=digest,
                )
                run.plan_draft = draft
                run.plan = plan
                run.usage.append(usage)
                next_job = job.model_copy(update={
                    "status": "awaiting_plan_review", "plan": plan,
                    "current_action": SynthesisAction(action_id=uuid4(), kind="approve_plan"),
                    "usage": list(run.usage),
                })
            elif stage == "extracting":
                if plan is None:
                    raise SynthesisStageError("plan_not_available")
                studies, usage = asyncio.run(self.staged_synthesis_engine.extract(papers, plan))
                run.studies = studies
                run.usage.append(usage)
                next_job = job.model_copy(update={
                    "status": "awaiting_study_review", "studies": studies,
                    "current_action": SynthesisAction(action_id=uuid4(), kind="approve_cards"),
                    "usage": list(run.usage),
                })
            elif stage == "synthesizing":
                if plan is None or len(studies) < 2:
                    raise SynthesisStageError("studies_not_available")
                from gene_evidence_agent.staged_synthesis import english_independence_notes
                notes = english_independence_notes(studies)
                analysis, catalog, usage = asyncio.run(
                    self.staged_synthesis_engine.synthesize(papers, studies, plan, notes)
                )
                run.analysis, run.catalog, run.independence_notes = analysis, catalog, notes
                run.usage.append(usage)
                report = build_report(
                    result_id, report_prepared, status="completed", plan=plan, studies=studies,
                    analysis=analysis, catalog=catalog, notes=notes, usages=run.usage,
                    synthesis_id=uuid4(), translation_status="not_offered",
                    warnings=analysis.warnings,
                )
                run.job = run.job.model_copy(update={
                    "status": "completed", "report": report,
                    "current_action": None,
                    "usage": list(run.usage),
                })
                self.synthesis_reports[report.synthesis_id] = report
                next_job = run.job
            elif stage == "translating":
                if job.report is None:
                    raise SynthesisStageError("report_not_available")
                translated, usage = asyncio.run(self.staged_synthesis_engine.translate(job.report))
                run.usage.append(usage)
                report = job.report.model_copy(update={
                    "chinese_summary": translated, "translation_status": "completed", "usage": list(run.usage),
                })
                self.synthesis_reports[report.synthesis_id] = report
                next_job = job.model_copy(update={
                    "status": "completed", "report": report, "current_action": None,
                    "usage": list(run.usage),
                })
            else:
                raise SynthesisStageError("invalid_stage")
            with self.lock:
                run = self.staged_synthesis_runs.get(key)
                if run is not None:
                    run.job = next_job
                    if source:
                        self.reports[result_id] = (self.clock(), self.reports[result_id][1])
        except Exception as error:
            if isinstance(error, SynthesisStageError):
                code = str(error)
            elif isinstance(error, (ValidationError, json.JSONDecodeError)):
                code = "invalid_model_output"
            elif isinstance(error, ModelTokenizerError):
                code = "model_tokenizer_unavailable"
            else:
                code = "internal_error"
            last_usage = self.staged_synthesis_engine.last_usage
            with self.lock:
                run = self.staged_synthesis_runs.get(key)
                if run is not None:
                    if last_usage and (not run.usage or run.usage[-1] != last_usage):
                        run.usage.append(last_usage)
                    if stage == "translating" and run.job.report is not None:
                        report = run.job.report.model_copy(update={
                            "translation_status": "failed", "usage": list(run.usage),
                        })
                        self.synthesis_reports[report.synthesis_id] = report
                        run.job = run.job.model_copy(update={
                            "status": "partial", "report": report, "error_code": code,
                            "current_action": None, "usage": list(run.usage),
                        })
                    else:
                        report = build_report(
                            result_id, report_prepared, status="partial" if run.studies else "failed",
                            plan=run.plan, studies=run.studies, analysis=run.analysis,
                            catalog=run.catalog, notes=run.independence_notes, usages=run.usage,
                            error_code=code,
                        )
                        run.job = run.job.model_copy(update={
                            "status": "partial" if run.studies or run.plan else "failed",
                            "report": report, "error_code": code,
                            "current_action": None, "usage": list(run.usage),
                        })
                    if self.staged_synthesis_engine.stop_on_failure:
                        self.staged_synthesis_engine.validation_failed = True
        finally:
            with self.lock:
                self.active = max(0, self.active - 1)

    def submit_synthesis(self, result_id):
        with self.lock:
            self._purge()
            if result_id not in self.reports:
                raise ChatServiceError("report_not_found", 404)
            if result_id in self.synthesis_jobs:
                return self.synthesis_jobs[result_id]
            if not self.synthesis_engine.health()["enabled"]:
                raise ChatServiceError("synthesis_disabled", 503)
            if self.active >= MAX_IN_FLIGHT:
                raise ChatServiceError("capacity_reached", 429)
            job = SynthesisJob(job_id=uuid4(), result_id=result_id, status="queued")
            self.synthesis_jobs[result_id] = job
            report = deepcopy(self.reports[result_id][1])
            self.active += 1
            self.executor.submit(self._run_synthesis, job, report)
            return job

    def get_synthesis_job(self, result_id, job_id):
        with self.lock:
            self._purge()
            staged = self.staged_synthesis_runs.get((result_id, job_id))
            if staged is not None:
                return staged.job
            job = self.synthesis_jobs.get(result_id)
            if job is None or job.job_id != job_id:
                raise ChatServiceError("job_not_found", 404)
            return job

    def _run_synthesis(self, job, report):
        def progress(status):
            with self.lock:
                self.synthesis_jobs[job.result_id] = job.model_copy(update={"status": status})
        try:
            result = self.synthesis_engine(job.result_id, report, progress)
            finished = job.model_copy(update={"status": result.status, "report": result, "error_code": result.error_code})
        except Exception:
            # Never expose exception text, model bodies or credentials.
            finished = job.model_copy(update={"status": "failed", "error_code": "internal_error"})
            if self.synthesis_engine.stop_on_failure:
                self.synthesis_engine.validation_failed = True
        with self.lock:
            self.synthesis_jobs[job.result_id] = finished
            if finished.report is not None:
                self.synthesis_reports[finished.report.synthesis_id] = finished.report
            if job.result_id in self.reports:
                self.reports[job.result_id] = (self.clock(), self.reports[job.result_id][1])
            self.active -= 1

    def clear(self, session_id: UUID):
        with self.lock:
            session = self._get_session(session_id)
            if session.busy:
                raise ChatServiceError("session_busy", 409)
            del self.sessions[session_id]
            self.attachments.discard_session(session_id)
            return {"status": "cleared"}

    def upload_attachment(self, session_id, filename, encoded):
        with self.lock:
            session = self._get_session(session_id)
            if session.busy:
                raise ChatServiceError("session_busy", 409)
            try:
                return self.attachments.add(session_id, filename, encoded)
            except AttachmentError as error:
                raise ChatServiceError(str(error), 400) from None

    def _get_session(self, key):
        self._purge()
        if key not in self.sessions:
            raise ChatServiceError("session_not_found", 404)
        session = self.sessions[key]
        session.touched = self.clock()
        return session

    def get_job(self, session_id: UUID, job_id: UUID) -> ChatJob:
        with self.lock:
            session = self._get_session(session_id)
            job = next((item for item in session.jobs.values() if item.job_id == job_id), None)
            if job is None:
                raise ChatServiceError("job_not_found", 404)
            return job

    def submit(self, request: ChatMessageRequest, *, knowledge: KnowledgeStore | None = None,
               history_text: str | None = None) -> ChatJob:
        request_knowledge = knowledge or self.knowledge
        if not set(request.knowledge_ids).issubset({card.id for card in request_knowledge.cards}):
            raise ChatServiceError("knowledge_card_not_found", 400)
        try:
            ensure_no_sensitive_query(request.text)
        except InvalidInputError:
            raise ChatServiceError("sensitive_query_rejected") from None
        with self.lock:
            session = self._get_session(request.session_id)
            if request.message_id in session.requests:
                if session.requests[request.message_id] != request:
                    raise ChatServiceError("message_id_conflict", 409)
                return session.jobs[request.message_id]
            responder = self.synthesis_responder if request.synthesis_id is not None else self.responder
            if self.mode != "mock" and responder is None:
                raise ChatServiceError("model_disabled", 503)
            if session.busy:
                raise ChatServiceError("session_busy", 409)
            if len(session.requests) >= MAX_TURNS:
                raise ChatServiceError("session_turn_limit", 409)
            if self.active >= MAX_IN_FLIGHT:
                raise ChatServiceError("capacity_reached", 429)
            result_id = request.result_id if request.use_evidence else None
            if session.bound and session.result_id != result_id:
                raise ChatServiceError("context_change_requires_new_session", 409)
            if session.bound and session.synthesis_id != request.synthesis_id:
                raise ChatServiceError("context_change_requires_new_session", 409)
            report = None
            if result_id is not None:
                if result_id not in self.reports:
                    raise ChatServiceError("report_not_found", 404)
                report = deepcopy(self.reports[result_id][1])
            if request.synthesis_id is not None:
                if not self.health()["synthesis_enabled"]:
                    raise ChatServiceError("synthesis_disabled", 503)
                synthesis = self.synthesis_reports.get(request.synthesis_id)
                if synthesis is None or synthesis.result_id != result_id or synthesis.status not in {"completed", "mock"}:
                    raise ChatServiceError("synthesis_not_available", 409)
                report["_synthesis"] = synthesis.model_dump(mode="json")
            attachment_ids = list(dict.fromkeys([*session.attachment_ids, *request.attachment_ids]))
            try:
                documents = self.attachments.resolve(request.session_id, attachment_ids)
                for document in documents:
                    ensure_no_sensitive_query(document["text"])
            except AttachmentError as error:
                raise ChatServiceError(str(error), 400) from None
            except InvalidInputError:
                raise ChatServiceError("sensitive_query_rejected", 422) from None
            if documents:
                report = report or {}
                report["_attachments"] = documents
            job = ChatJob(job_id=uuid4(), session_id=request.session_id,
                          message_id=request.message_id, status="queued")
            session.bound, session.result_id, session.busy = True, result_id, True
            session.synthesis_id = request.synthesis_id
            session.attachment_ids = attachment_ids
            session.requests[request.message_id] = request
            session.jobs[request.message_id] = job
            history = list(session.history)
            self.active += 1
            self.executor.submit(self._run, request, job, history, report, responder, request_knowledge, history_text)
            return job

    def _run(self, request, job, history, report, responder, knowledge, history_text=None):
        with self.lock:
            session = self.sessions[request.session_id]
            session.jobs[request.message_id] = job.model_copy(update={"status": "running"})
        try:
            if self.mode == "mock":
                reply = ChatReply(status="mock", answer="模拟对话：已收到研究问题。这里将显示知识解释和来源引用。本回复未调用模型，也不是生物医学结论。",
                                  knowledge_version=knowledge.version, knowledge_matched=False,
                                  context_kind="mock", result_id=request.result_id if request.use_evidence else None,
                                  usage=ChatUsage(status="not_called"))
            else:
                reply = responder(request, history, report, knowledge)
                if not isinstance(reply, ChatReply):
                    raise ValueError("invalid responder result")
        except Exception as error:
            # Log only safe diagnostic metadata; never expose prompts, replies, or provider bodies.
            logger.warning(
                "Chat job failed unexpectedly mode=%s exception_type=%s",
                self.mode,
                type(error).__name__,
            )
            reply = ChatReply(status="failed", knowledge_version=knowledge.version,
                              knowledge_matched=False, context_kind="general", error_code="model_error",
                              result_id=request.result_id if request.use_evidence else None,
                              usage=ChatUsage(status="unknown" if self.mode == "live" else "not_called"))
        with self.lock:
            session = self.sessions[request.session_id]
            if request.synthesis_id is not None and reply.status == "failed" and self.synthesis_engine.stop_on_failure:
                self.synthesis_engine.validation_failed = True
            if reply.status != "failed":
                session.history.extend([("user", history_text or request.text), ("assistant", reply.answer)])
                session.history = session.history[-40:]
            session.jobs[request.message_id] = job.model_copy(update={
                "status": "failed" if reply.status == "failed" else "completed", "reply": reply})
            session.busy = False
            session.touched = self.clock()
            self.active -= 1
