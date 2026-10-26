"""Confirmation-gated conversation workflow for evidence retrieval and reports.

The interpreter is descriptive only. Retrieval, localization, and synthesis are
dispatchable only through a server-stored action ID explicitly confirmed by the
browser. Action state and result versions are bounded, process-local, and expire.
"""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic, sleep
from typing import Any, Callable, Literal
from uuid import UUID, uuid4

from pydantic import Field, ValidationError, field_validator

from gene_evidence_agent.chat import ChatServiceError, ResearchChatService
from gene_evidence_agent.chat_schemas import ChatMessageRequest
from gene_evidence_agent.disease_terms import (
    UnsupportedChineseDiseaseTerm,
    contains_chinese_text,
    normalize_disease_search_term,
)
from gene_evidence_agent.errors import InvalidInputError, ModelOutputError, ToolError, UpstreamError
from gene_evidence_agent.model_adapter import ModelAdapter, ModelMessage, OpenAICompatibleModelAdapter
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.schemas import StrictModel
from gene_evidence_agent.workflow_schemas import (
    WorkflowConfirmActionRequest,
    WorkflowBindSynthesisRequest,
    WorkflowMessageRequest,
    WorkflowPrepareActionRequest,
    WorkflowResolveEntityRequest,
    WorkflowSelectResultRequest,
)

SESSION_TTL_SECONDS = 1800
MAX_SESSIONS = 100
MAX_MESSAGES = 100
MAX_WORKERS = 4

WELCOME_MESSAGE = (
    "你好，我是基因—癌症证据研究助手，可以帮你检索公开资料，并查看多篇论文的研究对象、方法与发现。"
    "我们先确认研究对象。第一步：请告诉我基因和疾病，例如“TP53 与肺癌”。"
)

INTERPRETER_PROMPT = """你是一个受限的基因—癌症研究需求解析器，只返回严格JSON，不执行工具。
根据用户最新消息和当前状态返回唯一对象，字段只能是 intent、gene、disease、disease_scope。
intent 只能是 research_request、on_topic_question、off_topic、personal_medical_request。
gene/disease 缺失时返回null；只提取用户明确提供的实体，不猜测、不补全、不改写成特定亚型。
disease字段保留用户输入的原始语种；用户输入中文癌症名称时原样返回，后续确定性映射会在检索确认卡展示英文检索词。不要自行翻译或改成更具体的亚型。
disease_scope 只能是 broad、subtype、unclear；疾病范围宽泛时标 broad，未说明时 unclear。
个人诊断、个人检查结果解释、个体治疗或用药建议归 personal_medical_request。
无关主题归 off_topic；基因/癌症研究常识问题归 on_topic_question；包含基因—疾病检索需求归 research_request。
不要回答医学问题、提出检索、整理或综合动作，不要输出解释和Markdown。"""


class WorkflowInterpretation(StrictModel):
    intent: Literal["research_request", "on_topic_question", "off_topic", "personal_medical_request"]
    gene: str | None = Field(default=None, max_length=120)
    disease: str | None = Field(default=None, max_length=120)
    disease_scope: Literal["broad", "subtype", "unclear"] = "unclear"

    @field_validator("gene", "disease")
    @classmethod
    def normalize_entities(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("entity names must be nonblank and contain no control characters")
        return value


class GuidedWorkflowInterpreter:
    """A small JSON-only model call; it has no tool registry or dispatch capability."""

    def __init__(self, settings, *, adapter_factory: Callable[..., ModelAdapter] | None = None) -> None:
        self.settings = settings
        self.adapter_factory = adapter_factory or OpenAICompatibleModelAdapter

    async def interpret(self, text: str, *, context: str = "") -> WorkflowInterpretation:
        adapter = self.adapter_factory(
            self.settings,
            budget_scope="guided-workflow",
            max_scope_tokens=10_000_000,
            max_request_tokens=128_000,
        )
        try:
            response = await adapter.complete(
                [
                    ModelMessage(role="system", content=INTERPRETER_PROMPT),
                    ModelMessage(role="user", content=(context + "\n用户消息：" + text)[-8000:]),
                ],
                max_output_tokens=400,
            )
            if response.finish_reason and response.finish_reason.casefold() in {"length", "max_tokens"}:
                raise ModelOutputError("Workflow interpretation was truncated", source="guided_workflow", failure_code="output_truncated")
            try:
                payload = json.loads(response.text)
                return WorkflowInterpretation.model_validate(payload)
            except (ValueError, TypeError, ValidationError) as error:
                raise ModelOutputError("Workflow interpretation did not match its schema", source="guided_workflow", failure_code="invalid_model_output") from error
        except (InvalidInputError, ToolError):
            raise
        except Exception as error:
            raise UpstreamError("Workflow interpretation failed", source="guided_workflow") from error
        finally:
            await adapter.aclose()

    def __call__(self, text: str, *, context: str = "") -> WorkflowInterpretation:
        return asyncio.run(self.interpret(text, context=context))


@dataclass
class WorkflowAction:
    action_id: UUID
    kind: Literal["retrieve", "localize", "synthesize"]
    status: Literal["pending", "running", "completed", "failed", "cancelled", "stale"]
    params: dict[str, Any]
    created_at: float
    stage: str = "awaiting_confirmation"
    error_code: str | None = None
    result_id: str | None = None
    result_version: int | None = None
    result: dict[str, Any] | None = None
    chat_job: dict[str, Any] | None = None
    assistant_message: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "action_id": str(self.action_id),
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "params": deepcopy(self.params),
            "error_code": self.error_code,
            "result_id": self.result_id,
            "result_version": self.result_version,
            "result": deepcopy(self.result),
            "chat_job": deepcopy(self.chat_job),
            "assistant_message": self.assistant_message,
        }


@dataclass
class WorkflowSession:
    touched: float
    step: str = "collecting"
    gene: str | None = None
    disease: str | None = None
    disease_scope: str = "unclear"
    sources: list[str] = field(default_factory=lambda: ["pubmed", "uniprot", "opentargets"])
    include_drugs: bool = True
    pending_action_id: UUID | None = None
    last_action_id: UUID | None = None
    actions: dict[UUID, WorkflowAction] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    current_result_id: str | None = None
    current_synthesis_id: str | None = None
    chat_session_id: UUID | None = None
    chat_job_sessions: dict[UUID, UUID] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)
    message_results: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    message_requests: dict[UUID, WorkflowMessageRequest] = field(default_factory=dict)
    recorded_chat_jobs: set[UUID] = field(default_factory=set)
    version: int = 0
    busy: bool = False
    entity_choices: dict[str, dict[str, str | None]] = field(default_factory=dict)


class WorkflowError(Exception):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


class GuidedWorkflowService:
    """Owns conversation state and dispatches only explicitly confirmed actions."""

    def __init__(
        self,
        *,
        mode: str,
        chat_service: ResearchChatService,
        query_handler: Callable[..., dict[str, Any]],
        interpreter: Callable[..., WorkflowInterpretation] | None = None,
        localization_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        localization_enabled: bool = False,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.mode = mode
        self.chat_service = chat_service
        self.query_handler = query_handler
        self.interpreter = interpreter
        self.localization_handler = localization_handler
        self.localization_enabled = bool(localization_enabled)
        self.clock = clock
        self.lock = RLock()
        self.sessions: dict[UUID, WorkflowSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="guided-workflow")

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def _purge(self) -> None:
        cutoff = self.clock() - SESSION_TTL_SECONDS
        self.sessions = {
            key: session for key, session in self.sessions.items()
            if session.touched >= cutoff or any(action.status == "running" for action in session.actions.values())
        }

    def _session(self, session_id: UUID) -> WorkflowSession:
        self._purge()
        session = self.sessions.get(session_id)
        if session is None:
            raise WorkflowError("session_not_found", 404)
        session.touched = self.clock()
        return session

    def create_session(self) -> dict[str, Any]:
        with self.lock:
            self._purge()
            if len(self.sessions) >= MAX_SESSIONS:
                raise WorkflowError("capacity_reached", 429)
            chat = self.chat_service.create_session()
            key = UUID(chat["session_id"])
            session = WorkflowSession(touched=self.clock(), chat_session_id=key)
            session.messages.append({"role": "assistant", "text": WELCOME_MESSAGE})
            self.sessions[key] = session
            return self.state(key)

    def state(self, session_id: UUID) -> dict[str, Any]:
        with self.lock:
            session = self._session(session_id)
            pending = session.actions.get(session.pending_action_id) if session.pending_action_id else None
            latest_result = None
            if session.current_result_id:
                latest_result = next(
                    (item for item in reversed(session.results) if item["result_id"] == session.current_result_id), None
                )
            choice = session.entity_choices.get(session.current_result_id or "")
            candidates = []
            if latest_result:
                candidates = [
                    {"id": entity.get("id"), "entity_type": entity.get("entity_type"),
                     "name": entity.get("canonical_name") or entity.get("query_name"),
                     "identifier": entity.get("identifier"), "namespace": entity.get("namespace"),
                     "resolution_status": entity.get("resolution_status")}
                    for entity in latest_result["payload"].get("entities", [])
                    if entity.get("entity_type") in {"gene", "disease"}
                    and entity.get("resolution_status") in {"resolved", "ambiguous"}
                ]
            return {
                "session_id": str(session_id),
                "step": session.step,
                "gene": session.gene,
                "disease": session.disease,
                "disease_scope": session.disease_scope,
                "sources": list(session.sources),
                "include_drugs": session.include_drugs,
                "pending_action": pending.public() if pending else None,
                "last_action": session.actions[session.last_action_id].public() if session.last_action_id in session.actions else None,
                "current_result": deepcopy(latest_result),
                "entity_resolution": {
                    "required": bool(latest_result and _has_ambiguity(latest_result["payload"]) and not choice),
                    "choice": deepcopy(choice),
                    "candidates": candidates,
                },
                "results": [
                    {"result_id": item["result_id"], "version": item["version"], "query": item["query"],
                     "synthesis_id": item.get("synthesis_id")}
                    for item in session.results
                ],
                "messages": deepcopy(session.messages),
                "capabilities": {
                    "retrieval": True,
                    "localization_enabled": False,
                    "synthesis_enabled": self.chat_service.staged_synthesis_engine.health()["enabled"],
                },
                "expires_in_seconds": SESSION_TTL_SECONDS,
            }

    def message(self, request: WorkflowMessageRequest) -> dict[str, Any]:
        try:
            ensure_no_sensitive_query(request.text)
        except InvalidInputError:
            raise WorkflowError("sensitive_query_rejected") from None
        with self.lock:
            session = self._session(request.session_id)
            previous = session.message_results.get(request.message_id)
            if previous is not None:
                if session.message_requests.get(request.message_id) != request:
                    raise WorkflowError("message_id_conflict", 409)
                return deepcopy(previous)
            if session.busy:
                raise WorkflowError("session_busy", 409)
            if len(session.messages) >= MAX_MESSAGES:
                raise WorkflowError("session_turn_limit", 409)
            session.busy = True
            if request.sources is not None:
                if request.include_drugs and "opentargets" not in request.sources:
                    session.busy = False
                    raise WorkflowError("drug_option_requires_opentargets")
                session.sources = list(request.sources)
            if request.include_drugs is not None:
                session.include_drugs = request.include_drugs
            session.messages.append({"role": "user", "text": request.text})
            step = session.step

        try:
            if (request.attachment_ids and not _is_personal_request(request.text.casefold())
                    and not _looks_like_new_research(request.text)):
                if session.current_result_id:
                    response = self._handle_followup(session, request)
                else:
                    response = self._submit_chat(session, request.text, request.attachment_ids)
            elif step == "collecting":
                interpreted = self._interpret(request.text, session)
                response = self._handle_collection(session, interpreted, request.text)
            elif step == "clarifying_scope":
                if _is_personal_request(request.text.casefold()):
                    response = self._reply(session, "我不能提供个人诊断、治疗或用药建议。" + self._current_prompt(session))
                elif not _is_research_topic(request.text.casefold()):
                    response = self._reply(session, "我专注于基因—癌症研究。" + self._current_prompt(session))
                else:
                    response = self._handle_scope(session, request.text)
            elif step == "awaiting_retrieval_confirmation":
                if _is_personal_request(request.text.casefold()):
                    response = self._reply(session, "我不能提供个人诊断、治疗或用药建议。" + self._current_prompt(session))
                elif not _is_research_topic(request.text.casefold()):
                    response = self._reply(session, "我专注于基因—癌症研究。" + self._current_prompt(session))
                else:
                    response = self._remind_confirmation(session)
            elif step == "resolving_entities":
                if _is_personal_request(request.text.casefold()):
                    response = self._reply(session, "我不能提供个人诊断、治疗或用药建议。请选择保留当前歧义，或在候选列表中指定关注对象。")
                elif any(token in request.text.casefold() for token in ("保留歧义", "保持歧义", "按原查询继续", "keep ambiguous")):
                    response = self._resolve_from_message(session)
                elif not _is_research_topic(request.text.casefold()):
                    response = self._reply(session, "我专注于基因—癌症研究。请选择保留当前歧义，或在候选列表中指定关注对象。")
                else:
                    response = self._reply(session, "请在下方候选列表中指定关注对象，或回复“保留歧义继续”。选择只记录你的分析关注点，不会把来源歧义伪装成已解析。")
            elif step in {"results_ready", "report_ready"}:
                if _looks_like_new_research(request.text):
                    interpreted = self._interpret(request.text, session)
                    session.gene = None
                    session.disease = None
                    response = self._handle_collection(session, interpreted, request.text)
                else:
                    response = self._handle_followup(session, request)
            else:
                response = self._reply(session, self._current_prompt(session))
        except WorkflowError:
            with self.lock:
                session.messages.pop()  # Do not retain a request that could not be processed.
                session.busy = False
            raise
        except Exception as error:
            with self.lock:
                session.messages.pop()
                session.busy = False
            raise WorkflowError("workflow_failed", 502) from error

        with self.lock:
            session = self._session(request.session_id)
            assistant = response.get("assistant_message") or response.get("message") or ""
            if assistant and not (session.messages and session.messages[-1] == {"role": "assistant", "text": assistant}):
                session.messages.append({"role": "assistant", "text": assistant})
            session.busy = False
            session.message_results[request.message_id] = deepcopy(response)
            session.message_requests[request.message_id] = request
            return response

    def _interpret(self, text: str, session: WorkflowSession) -> WorkflowInterpretation:
        if self.mode == "live" and self.interpreter is not None:
            try:
                return self.interpreter(
                    text,
                    context=f"步骤={session.step}; 已确认基因={session.gene or '未确认'}; 已确认疾病={session.disease or '未确认'}",
                )
            except ToolError as error:
                raise WorkflowError("interpretation_failed", 502) from error
        return deterministic_interpretation(text)

    def _handle_collection(self, session: WorkflowSession, intent: WorkflowInterpretation, text: str) -> dict[str, Any]:
        if intent.intent == "personal_medical_request":
            return self._reply(session, "我不能提供个人诊断、治疗或用药建议；可以帮你检索公开研究。" + self._current_prompt(session))
        if intent.intent == "off_topic":
            return self._reply(session, "我专注于基因—癌症公开证据研究。" + self._current_prompt(session))
        if intent.intent == "on_topic_question":
            return self._submit_chat(session, text)
        session.gene = intent.gene or session.gene
        session.disease = intent.disease or session.disease
        session.disease_scope = intent.disease_scope
        if not session.gene or not session.disease:
            missing = "基因" if not session.gene else "疾病"
            return self._reply(session, f"我还需要确认{missing}名称。{self._current_prompt(session)}")
        if intent.disease_scope in {"broad", "unclear"}:
            session.step = "clarifying_scope"
            return self._reply(
                session,
                f"我提取到：基因 {session.gene}，疾病 {session.disease}。疾病范围可能较宽。你希望按广义疾病检索，还是指定癌症亚型（如非小细胞肺癌）？",
            )
        return self._propose_retrieval(session)

    def _handle_scope(self, session: WorkflowSession, text: str) -> dict[str, Any]:
        normalized = text.casefold()
        broad = any(token in normalized for token in ("广义", "整体", "全部", "所有", "不限定", "按肺癌", "按乳腺癌", "broad", "overall"))
        subtype = not broad and any(token in normalized for token in ("亚型", "指定", "非小细胞", "小细胞", "腺癌", "鳞癌", "nsclc", "sclc", "adenocarcinoma"))
        if subtype:
            supplied = re.search(r"(?:改为|限定为|亚型(?:是|为)?|指定(?:为)?)[：:\s]*([^，,。；;]{2,80})", text)
            if supplied:
                session.disease = supplied.group(1).strip()
            elif any(token in normalized for token in ("非小细胞", "nsclc")):
                session.disease = "非小细胞肺癌"
            elif "小细胞" in text or "sclc" in normalized:
                session.disease = "小细胞肺癌"
            elif "腺癌" in text or "adenocarcinoma" in normalized:
                session.disease = "肺腺癌"
            session.disease_scope = "subtype"
            return self._propose_retrieval(session)
        if broad:
            session.disease_scope = "broad"
            return self._propose_retrieval(session)
        return self._reply(session, "请回复“按广义疾病检索”，或告诉我具体癌症亚型。")

    def _propose_retrieval(self, session: WorkflowSession) -> dict[str, Any]:
        if session.gene and contains_chinese_text(session.gene):
            session.step = "collecting"
            return self._reply(
                session,
                "数据库检索需要英文基因符号。当前没有启动检索；请重新发送完整条件，例如“TP53 与肺癌”。",
            )
        try:
            disease_term = normalize_disease_search_term(session.disease or "")
        except UnsupportedChineseDiseaseTerm:
            session.step = "collecting"
            return self._reply(
                session,
                "我还不能可靠地把这个中文疾病名称对应到英文检索词，当前没有启动检索。请重新发送完整的基因和英文疾病名称，例如“TP53 与 lung cancer”；也可以改用确认卡中支持的中文癌症名称。",
            )
        session.step = "awaiting_retrieval_confirmation"
        action = self._new_action(session, "retrieve", {
            "gene": session.gene,
            "disease": disease_term.search_term,
            "disease_input": disease_term.original,
            "sources": list(session.sources),
            "include_drugs": session.include_drugs,
        })
        translation_note = (
            f"检测到中文疾病名“{disease_term.original}”，已规范为英文检索词“{disease_term.search_term}”。"
            if disease_term.translated else ""
        )
        return self._reply(
            session,
            (translation_note + "请检查下方英文检索词及其他条件。默认选择 PubMed、UniProt、Open Targets 和疾病层级药物候选；点击“确认并检索”后才会查询。").strip(),
            action=action,
        )

    def prepare_action(self, request: WorkflowPrepareActionRequest) -> dict[str, Any]:
        with self.lock:
            session = self._session(request.session_id)
            if session.step not in {"results_ready", "report_ready"} or not session.current_result_id:
                raise WorkflowError("result_required", 409)
            selected = next((item for item in reversed(session.results) if item["result_id"] == session.current_result_id), None)
            if selected is None:
                raise WorkflowError("result_not_found", 404)
            if request.action == "synthesize":
                abstracts = _eligible_abstract_count(selected["payload"])
                if abstracts < 2:
                    raise WorkflowError("insufficient_abstracts", 409)
                if not self.chat_service.staged_synthesis_engine.health()["enabled"]:
                    raise WorkflowError("synthesis_disabled", 503)
                raise WorkflowError("staged_synthesis_required", 409)
            if request.action == "localize" and self.mode != "mock" and (
                self.localization_handler is None or not self.localization_enabled
            ):
                raise WorkflowError("localization_disabled", 503)
            if session.pending_action_id:
                pending = session.actions.get(session.pending_action_id)
                if pending and pending.status in {"pending", "running"}:
                    raise WorkflowError("operation_in_progress", 409)
            action = self._new_action(session, request.action, {"result_id": session.current_result_id})
            action.stage = "awaiting_confirmation"
            text = _action_confirmation_text(request.action, mode=self.mode)
            session.messages.append({"role": "assistant", "text": text})
            return self._reply(session, text, action=action)

    def confirm_action(self, request: WorkflowConfirmActionRequest) -> dict[str, Any]:
        with self.lock:
            session = self._session(request.session_id)
            action = session.actions.get(request.action_id)
            if action is None:
                raise WorkflowError("action_stale", 409)
            if action.status in {"completed", "failed", "cancelled"}:
                return self._action_reply(session, action)
            if session.pending_action_id != request.action_id or action.status == "stale":
                raise WorkflowError("action_stale", 409)
            if action.status != "pending":
                return self._action_reply(session, action)
            changed = False
            if action.kind == "retrieve":
                sources = request.sources if request.sources is not None else action.params["sources"]
                drugs = request.include_drugs if request.include_drugs is not None else action.params["include_drugs"]
                if drugs and "opentargets" not in sources:
                    raise WorkflowError("drug_option_requires_opentargets")
                gene = action.params["gene"]
                disease = action.params["disease"]
                # Editable entities are accepted only as confirmation-card values, then
                # receive a new action ID so the original confirmation cannot authorize them.
                gene = request.gene or gene
                submitted_disease = request.disease or disease
                disease_input = (
                    action.params.get("disease_input", disease)
                    if submitted_disease == disease else submitted_disease
                )
                if request.decision == "confirm":
                    if contains_chinese_text(gene):
                        raise WorkflowError("english_gene_symbol_required", 400)
                    try:
                        disease_term = normalize_disease_search_term(disease_input)
                    except UnsupportedChineseDiseaseTerm as error:
                        raise WorkflowError("unsupported_chinese_disease_term", 400) from error
                    disease = disease_term.search_term
                changed = (request.decision == "confirm" and (list(sources) != action.params["sources"] or drugs != action.params["include_drugs"]
                           or gene != action.params["gene"] or disease != action.params["disease"])
                           )
                if changed:
                    if not isinstance(gene, str) or not gene.strip() or len(gene) > 120 or not isinstance(disease, str) or not disease.strip() or len(disease) > 120:
                        raise WorkflowError("invalid_query_field")
                    action.status = "stale"
                    action.stage = "parameters_changed_confirmation_required"
                    action = self._new_action(session, "retrieve", {
                        "gene": gene.strip(), "disease": disease.strip(), "disease_input": disease_input.strip(),
                        "sources": list(sources), "include_drugs": drugs,
                    })
                    session.gene, session.disease = gene.strip(), disease_input.strip()
                    notice = "检索条件已修改；旧确认已失效。请核对新确认卡并再次点击确认。"
                    if disease_term.translated:
                        notice = f"中文疾病名已规范为英文检索词“{disease_term.search_term}”。" + notice
                    session.messages.append({"role": "assistant", "text": notice})
                    return self._reply(session, notice, action=action)
                if request.decision == "cancel":
                    action.status, action.stage = "cancelled", "cancelled_by_user"
                    session.pending_action_id = None
                    session.step = "collecting"
                    notice = "已取消检索。你可以继续修改基因或疾病条件。"
                    session.messages.append({"role": "assistant", "text": notice})
                    return self._action_reply(session, action, notice)
            elif request.decision == "cancel":
                action.status, action.stage = "cancelled", "cancelled_by_user"
                session.pending_action_id = None
                session.step = "results_ready" if session.current_result_id else "collecting"
                notice = "已取消该操作；当前检索结果仍保留。"
                session.messages.append({"role": "assistant", "text": notice})
                return self._action_reply(session, action, notice)
            action.status = "running"
            action.stage = "queued"
            action.assistant_message = "已确认，正在执行。只显示服务端实际状态，不展示虚构进度。"
            session.messages.append({"role": "assistant", "text": action.assistant_message})
            self.executor.submit(self._run_action, request.session_id, action.action_id)
            return self._action_reply(session, action, action.assistant_message)

    def select_result(self, request: WorkflowSelectResultRequest) -> dict[str, Any]:
        with self.lock:
            session = self._session(request.session_id)
            selected = next((item for item in session.results if item["result_id"] == str(request.result_id)), None)
            if selected is None:
                raise WorkflowError("result_not_found", 404)
            if session.pending_action_id:
                pending = session.actions.get(session.pending_action_id)
                if pending and pending.status in {"pending", "running"}:
                    raise WorkflowError("operation_in_progress", 409)
            session.current_result_id = selected["result_id"]
            session.current_synthesis_id = selected.get("synthesis_id")
            choice = session.entity_choices.get(selected["result_id"])
            session.step = "resolving_entities" if _has_ambiguity(selected["payload"]) and not choice else "report_ready" if session.current_synthesis_id else "results_ready"
            self._reset_chat_context(session)
            return self.state(request.session_id)

    def bind_synthesis(self, request: WorkflowBindSynthesisRequest) -> dict[str, Any]:
        """Bind a completed staged report to the current evidence version for follow-up chat."""
        with self.lock:
            session = self._session(request.session_id)
            result_id = str(request.result_id)
            if session.current_result_id != result_id:
                raise WorkflowError("result_version_changed", 409)
            entry = next((item for item in reversed(session.results) if item["result_id"] == result_id), None)
            if entry is None:
                raise WorkflowError("result_not_found", 404)
            report = self.chat_service.synthesis_reports.get(request.synthesis_id)
            if report is None or str(report.result_id) != result_id or report.status != "completed":
                raise WorkflowError("synthesis_not_available", 409)
            if session.current_synthesis_id == str(request.synthesis_id) and entry.get("synthesis_id") == str(request.synthesis_id):
                return self.state(request.session_id)
            session.current_synthesis_id = str(request.synthesis_id)
            entry["synthesis_id"] = session.current_synthesis_id
            session.step = "report_ready"
            self._reset_chat_context(session)
            notice = "英文跨文献报告已绑定到当前检索版本；可以基于报告和原始摘要继续追问。综合推断与研究假设需结合引用核对，不能作为已证实结论。"
            session.messages.append({"role": "assistant", "text": notice})
            return {**self.state(request.session_id), "assistant_message": notice}

    def _reset_chat_context(self, session: WorkflowSession) -> None:
        old_session_id = session.chat_session_id
        if old_session_id is not None:
            try:
                self.chat_service.clear(old_session_id)
            except ChatServiceError:
                # A pending job retains its original context until it settles; the
                # job-to-session map below keeps polling bound to that old session.
                pass
        session.chat_session_id = UUID(self.chat_service.create_session()["session_id"])

    def resolve_entity(self, request: WorkflowResolveEntityRequest) -> dict[str, Any]:
        with self.lock:
            session = self._session(request.session_id)
            result = next((item for item in session.results if item["result_id"] == str(request.result_id)), None)
            if result is None or session.current_result_id != result["result_id"]:
                raise WorkflowError("result_not_found", 404)
            if not _has_ambiguity(result["payload"]):
                raise WorkflowError("no_entity_ambiguity", 409)
            entity_id = request.entity_id
            if request.decision == "select_candidate":
                match = next((entity for entity in result["payload"].get("entities", [])
                              if entity.get("id") == entity_id and entity.get("entity_type") in {"gene", "disease"}
                              and entity.get("resolution_status") in {"resolved", "ambiguous"}), None)
                if match is None:
                    raise WorkflowError("candidate_not_in_result", 400)
                choice = {"decision": "selected_focus", "entity_id": entity_id,
                          "name": match.get("canonical_name") or match.get("query_name"),
                          "identifier": match.get("identifier"), "namespace": match.get("namespace")}
                notice = "已记录你选择的关注候选；原始来源歧义和所有证据均保留，这不等同于身份已被确认。"
            else:
                choice = {"decision": "kept_ambiguous", "entity_id": None, "name": None,
                          "identifier": None, "namespace": None}
                notice = "已保留实体歧义并继续；后续整理不会把候选当作唯一实体。"
            session.entity_choices[result["result_id"]] = choice
            session.step = "report_ready" if result.get("synthesis_id") else "results_ready"
            session.messages.append({"role": "assistant", "text": notice})
            return {**self.state(request.session_id), "assistant_message": notice}

    def _resolve_from_message(self, session: WorkflowSession) -> dict[str, Any]:
        if not session.current_result_id:
            raise WorkflowError("result_not_found", 404)
        request = WorkflowResolveEntityRequest(
            session_id=next(key for key, value in self.sessions.items() if value is session),
            result_id=UUID(session.current_result_id), decision="keep_ambiguous",
        )
        result = self.resolve_entity(request)
        return {key: result.get(key) for key in ("session_id", "step", "assistant_message", "action", "chat_job")}

    def get_chat_job(self, session_id: UUID, job_id: UUID) -> dict[str, Any]:
        with self.lock:
            session = self._session(session_id)
            try:
                owner_id = session.chat_job_sessions.get(job_id, session.chat_session_id)
                job = self.chat_service.get_job(owner_id, job_id)
            except ChatServiceError as error:
                raise WorkflowError(error.code, error.status) from error
            if job.reply is not None and job.status == "completed" and job.job_id not in session.recorded_chat_jobs:
                session.messages.append({"role": "assistant", "text": job.reply.answer})
                session.recorded_chat_jobs.add(job.job_id)
            return job.model_dump(mode="json")

    def _run_action(self, session_id: UUID, action_id: UUID) -> None:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            action = session.actions.get(action_id)
            if action is None or action.status != "running":
                return
            action.stage = "running"
            params = deepcopy(action.params)
            snapshot = next((item for item in reversed(session.results) if item["result_id"] == params.get("result_id")), None)
            payload = deepcopy(snapshot["payload"]) if snapshot else None
        try:
            if action.kind == "retrieve":
                payload = self.query_handler(
                    params["gene"], params.get("disease_input", params["disease"]),
                    set(params["sources"]), params["include_drugs"]
                )
                payload = deepcopy(payload)
                result_id = self.chat_service.register_report(payload)
                payload["result_id"] = result_id
                with self.lock:
                    session = self._session(session_id)
                    session.version += 1
                    entry = {"result_id": result_id, "version": session.version,
                             "query": f"{params['gene']} 与 {params['disease']}", "gene": params["gene"], "disease": params["disease"], "payload": payload,
                             "synthesis_id": None}
                    session.results.append(entry)
                    session.current_result_id = result_id
                    session.current_synthesis_id = None
                    session.step = "resolving_entities" if _has_ambiguity(payload) else "results_ready"
                    self._reset_chat_context(session)
                message = _result_message(payload)
            elif action.kind == "localize":
                if payload is None:
                    raise WorkflowError("result_not_found", 404)
                if self.mode == "mock":
                    localized = {"translation_status": "model_disabled", "overview_zh": "模拟模式未调用模型；英文原始证据保留。", "translated_record_count": 0}
                else:
                    localized = self.localization_handler(payload) if self.localization_handler else None
                    if localized is None:
                        raise WorkflowError("localization_disabled", 503)
                updated = {**payload, "chinese_report": localized}
                result_id = self.chat_service.register_report(updated)
                updated["result_id"] = result_id
                payload = updated
                with self.lock:
                    session = self._session(session_id)
                    session.version += 1
                    prior_choice = session.entity_choices.get(str(params["result_id"]))
                    entry = {"result_id": result_id, "version": session.version,
                             "query": session.results[-1]["query"], "gene": session.results[-1]["gene"],
                             "disease": session.results[-1]["disease"], "payload": updated,
                             "synthesis_id": None}
                    session.results.append(entry)
                    if prior_choice:
                        session.entity_choices[result_id] = deepcopy(prior_choice)
                    session.current_result_id = result_id
                    session.current_synthesis_id = None
                    session.step = "resolving_entities" if _has_ambiguity(updated) and not prior_choice else "results_ready"
                    self._reset_chat_context(session)
                message = "中文整理操作已结束。模型整理与英文原文分开展示；请核对来源原句。你也可以单独生成逐篇文献对照报告。"
            else:
                if payload is None:
                    raise WorkflowError("result_not_found", 404)
                result_uuid = UUID(params["result_id"])
                job = self.chat_service.submit_synthesis(result_uuid)
                while job.status in {"queued", "extracting", "synthesizing"}:
                    sleep(0.2)
                    job = self.chat_service.get_synthesis_job(result_uuid, job.job_id)
                if job.report is None:
                    raise WorkflowError(job.error_code or "synthesis_failed", 502)
                message = "逐篇结构化对照报告已完成并绑定到本次检索版本；可继续追问报告中的单篇研究事实。"
                with self.lock:
                    session = self._session(session_id)
                    session.current_synthesis_id = str(job.report.synthesis_id)
                    entry = next((item for item in reversed(session.results) if item["result_id"] == params["result_id"]), None)
                    if entry is not None:
                        entry["synthesis_id"] = session.current_synthesis_id
                    session.step = "report_ready"
                    self._reset_chat_context(session)
                action.result = {"synthesis": job.report.model_dump(mode="json")}
            with self.lock:
                session = self._session(session_id)
                action = session.actions[action_id]
                action.status, action.stage = "completed", "completed"
                action.result_id = session.current_result_id
                action.result_version = session.version
                action.result = action.result or deepcopy(payload)
                action.assistant_message = message
                if message:
                    session.messages.append({"role": "assistant", "text": message})
                if session.pending_action_id == action_id:
                    session.pending_action_id = None
        except Exception as error:
            with self.lock:
                session = self.sessions.get(session_id)
                if session is None:
                    return
                action = session.actions[action_id]
                action.status = "failed"
                action.stage = "failed"
                action.error_code = error.code if isinstance(error, WorkflowError) else "operation_failed"
                action.assistant_message = "该操作未能完成；已保留之前的检索结果。没有自动重试。"
                session.messages.append({"role": "assistant", "text": action.assistant_message})
                if session.pending_action_id == action_id:
                    session.pending_action_id = None
                if action.kind == "retrieve":
                    session.step = "collecting"
                else:
                    session.step = "results_ready"

    def _handle_followup(self, session: WorkflowSession, request: WorkflowMessageRequest) -> dict[str, Any]:
        text = request.text.casefold()
        if _is_personal_request(text):
            return self._reply(session, "我不能提供个人诊断、治疗或用药建议；可以继续讨论公开研究证据。" + self._next_action_prompt(session))
        if not request.attachment_ids and not _is_research_topic(text):
            return self._reply(session, "我专注于基因—癌症研究。" + self._next_action_prompt(session))
        try:
            result_id = UUID(session.current_result_id) if session.current_result_id else None
            request_to_chat = ChatMessageRequest(
                session_id=session.chat_session_id,
                message_id=request.message_id,
                text=(f"用户将 {session.entity_choices[session.current_result_id]['name']} 作为关注候选；原报告仍有实体歧义，不能视作已解析。用户问题："
                      if session.current_result_id in session.entity_choices and session.entity_choices[session.current_result_id].get("decision") == "selected_focus" else "") + request.text,
                result_id=result_id,
                synthesis_id=UUID(session.current_synthesis_id) if session.current_synthesis_id else None,
                use_evidence=result_id is not None,
                attachment_ids=request.attachment_ids,
            )
            job = self.chat_service.submit(request_to_chat)
        except ChatServiceError as error:
            raise WorkflowError(error.code, error.status) from error
        session.chat_job_sessions[job.job_id] = session.chat_session_id
        return self._reply(session, "问题已交给当前结果版本的研究对话处理；来源引用将保留。", chat_job=job.model_dump(mode="json"))

    def _submit_chat(self, session: WorkflowSession, request_text: str, attachment_ids=None) -> dict[str, Any]:
        try:
            job = self.chat_service.submit(ChatMessageRequest(
                session_id=session.chat_session_id,
                message_id=uuid4(),
                text=request_text,
                use_evidence=False,
                attachment_ids=attachment_ids or [],
            ))
        except ChatServiceError as error:
            raise WorkflowError(error.code, error.status) from error
        session.chat_job_sessions[job.job_id] = session.chat_session_id
        return self._reply(session, "这是基因—癌症研究问题，正在整理有来源的简要回答。完成后我会回到当前待办步骤。", chat_job=job.model_dump(mode="json"))

    def _remind_confirmation(self, session: WorkflowSession) -> dict[str, Any]:
        return self._reply(session, "检索尚未执行。请在确认卡中检查基因、疾病、来源与药物选项，再点击“确认并检索”；也可直接修改卡片条件。")

    def _current_prompt(self, session: WorkflowSession) -> str:
        if session.step == "clarifying_scope":
            return "你希望按广义疾病检索，还是指定癌症亚型？"
        if session.step == "awaiting_retrieval_confirmation":
            return "请在检索确认卡中核对条件，并点击“确认并检索”；也可以修改卡片内容。"
        if session.step == "resolving_entities":
            return "请在候选列表中指定关注对象，或选择“保留歧义继续”。"
        if not session.gene:
            return "第一步：请告诉我基因和疾病，例如“TP53 与肺癌”。"
        if not session.disease:
            return f"已记录基因 {session.gene}。请再告诉我想研究的疾病或癌症亚型。"
        if session.step in {"results_ready", "report_ready"}:
            return self._next_action_prompt(session)
        return f"当前研究对象是 {session.gene} 与 {session.disease}。"

    def _next_action_prompt(self, session: WorkflowSession) -> str:
        if session.step == "report_ready":
            return "你可以点击“追问当前报告”，或先切换到其他检索版本。"
        return "你可以查看英文证据、开始逐篇文献对照，或围绕当前结果提问。"

    def _new_action(self, session: WorkflowSession, kind: Literal["retrieve", "localize", "synthesize"], params: dict[str, Any]) -> WorkflowAction:
        if session.pending_action_id:
            prior = session.actions.get(session.pending_action_id)
            if prior and prior.status == "running":
                raise WorkflowError("operation_in_progress", 409)
            if prior and prior.status == "pending":
                prior.status, prior.stage = "stale", "superseded"
        action = WorkflowAction(action_id=uuid4(), kind=kind, status="pending", params=deepcopy(params), created_at=self.clock())
        session.actions[action.action_id] = action
        session.pending_action_id = action.action_id
        session.last_action_id = action.action_id
        return action

    def _reply(self, session: WorkflowSession, text: str, *, action: WorkflowAction | None = None, chat_job: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "session_id": str(next((key for key, value in self.sessions.items() if value is session), "")),
            "step": session.step,
            "assistant_message": text,
            "action": action.public() if action else None,
            "chat_job": chat_job,
        }

    def _action_reply(self, session: WorkflowSession, action: WorkflowAction, text: str | None = None) -> dict[str, Any]:
        return {
            "session_id": str(next((key for key, value in self.sessions.items() if value is session), "")),
            "step": session.step,
            "assistant_message": text or action.assistant_message or "",
            "action": action.public(),
            "chat_job": None,
        }


def deterministic_interpretation(text: str) -> WorkflowInterpretation:
    """Offline fallback used only when the workflow model is not enabled."""
    normalized = text.casefold()
    if _is_personal_request(normalized):
        return WorkflowInterpretation(intent="personal_medical_request")
    if not _is_research_topic(normalized) and not re.search(r"[A-Z][A-Z0-9-]{1,14}", text):
        return WorkflowInterpretation(intent="off_topic")
    gene = None
    disease = None
    match = re.search(r"\b([A-Z][A-Z0-9-]{1,14})\b\s*(?:与|和|关联(?:于)?|对应(?:于)?|and|vs\.?|[-—])\s*(.{2,120})", text, re.I)
    if match:
        gene = match.group(1).strip()
        disease = _clean_disease(match.group(2))
    else:
        gene_match = re.search(r"\b([A-Z][A-Z0-9-]{1,14})\b", text)
        if gene_match:
            gene = gene_match.group(1)
        disease_match = re.search(r"(?:肺癌|肺腺癌|非小细胞肺癌|小细胞肺癌|乳腺癌|结直肠癌|前列腺癌|卵巢癌|肝癌|胃癌|lung cancer|breast cancer|colorectal cancer|prostate cancer|ovarian cancer)", text, re.I)
        if disease_match:
            disease = disease_match.group(0)
    has_retrieval_verb = any(token in normalized for token in ("检索", "查询", "查找", "研究", "证据", "关系", "association", "search", "evidence"))
    if not gene and not disease and not has_retrieval_verb:
        return WorkflowInterpretation(intent="on_topic_question")
    if gene and not disease and re.search(r"(是什么|什么作用|有什么作用|功能|介绍|what is|function)", normalized, re.I) and not has_retrieval_verb:
        return WorkflowInterpretation(intent="on_topic_question")
    scope = "broad" if disease and any(token in disease.casefold() for token in ("肺癌", "lung cancer", "乳腺癌", "breast cancer", "结直肠癌", "colorectal cancer")) else "subtype" if disease else "unclear"
    return WorkflowInterpretation(intent="research_request", gene=gene, disease=disease, disease_scope=scope)


def _clean_disease(value: str) -> str:
    cleaned = re.split(r"[，,。；;?？!！\n]", value, maxsplit=1)[0]
    # Longer suffixes must be removed before the generic "关系" token; otherwise
    # inputs such as "肺癌的关系" leave the dangling character "的" behind.
    for prefix in ("之间的关系", "间的关系", "之间关系", "间关系", "的关系", "中的作用", "的作用", "关系", "evidence for", "relationship between"):
        cleaned = re.sub(rf"\s*{re.escape(prefix)}.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:与|和|关于|查询|检索|研究)\s*", "", cleaned)
    return cleaned.strip(" \t：:，,。；;")[:120]


def _is_personal_request(text: str) -> bool:
    return bool(re.search(r"(我(?:患有|得了|被诊断|的检查|的基因检测|该吃|该用|能不能用)|我.*(?:癌症|肿瘤).*(?:治疗|药|用药)|个人(?:诊断|治疗|用药)|给我开药|替我诊断|my\s+(?:diagnosis|test|cancer|treatment|medication)|should\s+i\s+(?:take|use))", text, re.I))


def _is_research_topic(text: str) -> bool:
    return bool(re.search(
        r"(基因|癌症|肿瘤|突变|靶点|文献|论文|摘要|原文|证据|药物|治疗机制|解释|比较|当前|结果|报告|这份|这个|为什么|差异|广义|亚型|检索|查询|研究设计|研究方法|研究对象|主要发现|研究发现|队列|样本|人群|癌症亚型|pmid|doi|tp53|brca|kras|egfr|nsclc|gene|cancer|tumou?r|mutation|pubmed|opentargets|evidence|paper|study|studies|abstract|finding|findings|design|method|population|compare|report|result|search|retrieve)",
        text,
        re.I,
    ))


def _looks_like_new_research(text: str) -> bool:
    normalized = text.strip().casefold()
    new_request = re.search(
        r"(?:"
        r"我(?:还)?想(?:要)?\s*(?:重新|再次|再)?\s*(?:检索|查询|查找|搜索|查|研究)|"
        r"我(?:要|希望)\s*(?:重新|再次|再)?\s*(?:检索|查询|查找|搜索|查|研究)|"
        r"(?:帮我|请帮我)\s*(?:重新|再次|再)?\s*(?:检索|查询|查找|搜索|查)|"
        r"(?:重新|再次|再一次|另外|新增|改查|改为)\s*(?:检索|查询|查找|搜索|查|研究)|"
        r"^\s*(?:请\s*)?(?:检索|查询|查找|搜索|查)(?:一下|一遍|新)?"
        r")",
        normalized,
    )
    return bool(
        (new_request or re.search(r"\b(?:search(?:\s+for)?|retrieve\s+new)\b", normalized))
        and re.search(r"[A-Z][A-Z0-9-]{1,14}", text)
        and _is_research_topic(text.casefold())
    )


def _eligible_abstract_count(result: dict[str, Any]) -> int:
    seen: set[str] = set()
    count = 0
    for record in result.get("evidence", []):
        source = record.get("source", {})
        if record.get("evidence_type") != "publication" or source.get("source_database") != "PubMed":
            continue
        abstract = record.get("abstract") or record.get("summary")
        if not abstract or "abstract_not_available" in record.get("warnings", []):
            continue
        key = str(source.get("record_id") or "").strip() or str(source.get("doi") or "").casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        count += 1
    return count


def _has_ambiguity(result: dict[str, Any]) -> bool:
    return any(item.get("status") == "ambiguous" for item in result.get("source_outcomes", [])) or any(
        entity.get("resolution_status") == "ambiguous" for entity in result.get("entities", [])
    )


def _action_confirmation_text(kind: str, *, mode: str = "live") -> str:
    if kind == "localize" and mode == "mock":
        return "当前为模拟模式。确认仅演示整理流程，不会调用模型；英文原始证据保持不变。"
    return {
        "localize": "请确认：将使用已检索证据调用模型生成中文整理。英文原文不会覆盖；模型失败时仍显示原文。",
        "synthesize": "请确认：将仅依据本次检索的 PubMed 摘要生成逐篇英文研究卡，并在本地组装并列对照报告；不生成跨篇推断。",
    }[kind]


def _result_message(result: dict[str, Any]) -> str:
    outcomes = result.get("source_outcomes", [])
    labels = {"success": "成功", "empty": "无结果", "ambiguous": "多个候选", "not_requested": "未选择"}
    source_text = "、".join(f"{item.get('source')}: {labels.get(item.get('status'), item.get('status', '未知'))}" for item in outcomes)
    ambiguous = any(item.get("status") == "ambiguous" for item in outcomes) or any(
        entity.get("resolution_status") == "ambiguous" for entity in result.get("entities", [])
    )
    note = " 部分实体有多个候选；请在左侧查看标识符，可选择保留歧义。" if ambiguous else ""
    return f"检索已结束。来源状态：{source_text or '未返回状态'}。结果保留原始记录和来源链接。{note}可进行逐篇文献对照；至少需要两篇英文摘要。"

