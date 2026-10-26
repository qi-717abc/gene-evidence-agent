"""Loopback-only web shell with a deterministic mock API for the staged UI build.

This phase intentionally does not load application settings, model credentials,
database clients, or the model adapter. Real retrieval is added in P3.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import asyncio
import hashlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4
from urllib.parse import urlsplit, parse_qs
from pydantic import ValidationError
from gene_evidence_agent import __version__
from gene_evidence_agent.attachments import UploadRequest, MAX_UPLOAD_BODY

from gene_evidence_agent.chat import ResearchChatService, ChatServiceError
from gene_evidence_agent.chat_schemas import ChatSessionRequest, ChatMessageRequest, ChatClearRequest
from gene_evidence_agent.education import (
    EducationCatalog, EducationChat, EducationClearRequest, EducationMessageRequest,
    EducationSessionRequest, EDUCATION_OUTPUT_CAP, EDUCATION_SCOPE, EDUCATION_STAGE_CAP,
    EDUCATION_SYSTEM_PROMPT,
)
from gene_evidence_agent.disease_terms import (
    UnsupportedChineseDiseaseTerm,
    contains_chinese_text,
    normalize_disease_search_term,
)
from gene_evidence_agent.synthesis import SynthesisEngine, SYNTHESIS_SCOPE
from gene_evidence_agent.synthesis_schemas import SynthesisRequest
from gene_evidence_agent.staged_synthesis import EnglishSynthesisEngine, SynthesisActionRequest

from gene_evidence_agent.errors import InvalidInputError
from gene_evidence_agent.config import Settings
from gene_evidence_agent.logging_config import configure_logging
from gene_evidence_agent.pipeline import GeneEvidencePipeline
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.schemas import BiomedicalEntity
from gene_evidence_agent.model_usage import (
    MODEL_LIFETIME_TOKEN_BUDGET,
    MODEL_LOCALIZATION_TOKEN_BUDGET,
    MODEL_P6_E2E_TOKEN_BUDGET,
    MODEL_REPAIR_TOKEN_BUDGET,
    P5_LOCALIZATION_BUDGET_SCOPE,
    P6_E2E_BUDGET_SCOPE,
    REPAIR_E2E_BUDGET_SCOPE,
    ModelUsageLedger,
)
from gene_evidence_agent.workflow import GuidedWorkflowInterpreter, GuidedWorkflowService, WorkflowError
from gene_evidence_agent.workflow_schemas import (
    WorkflowConfirmActionRequest,
    WorkflowBindSynthesisRequest,
    WorkflowMessageRequest,
    WorkflowPrepareActionRequest,
    WorkflowResolveEntityRequest,
    WorkflowSelectResultRequest,
    WorkflowSessionRequest,
)

HOST = "127.0.0.1"
PORT = 8765
MAX_BODY_BYTES = 4096
WEB_ROOT = Path(__file__).with_name("web")
STATIC_FILES = {
    "/": WEB_ROOT / "index.html",
    "/index.html": WEB_ROOT / "index.html",
    "/styles.css": WEB_ROOT / "styles.css",
    "/app.js": WEB_ROOT / "app.js",
    "/chat.js": WEB_ROOT / "chat.js",
    "/attachments.js": WEB_ROOT / "attachments.js",
    "/synthesis.js": WEB_ROOT / "synthesis.js",
    "/workflow.js": WEB_ROOT / "workflow.js",
    "/education-preferences.js": WEB_ROOT / "education_preferences.js",
    "/education-learning.js": WEB_ROOT / "education_learning.js",
    "/education.js": WEB_ROOT / "education.js",
    "/education-onboarding.js": WEB_ROOT / "education_onboarding.js",
    "/education-guides/GUIDE-01-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-01-web.jpg",
    "/education-guides/GUIDE-02-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-02-web.jpg",
    "/education-guides/GUIDE-03-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-03-web.jpg",
    "/education-guides/GUIDE-04-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-04-web.jpg",
    "/education-guides/GUIDE-05-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-05-web.jpg",
    "/education-guides/GUIDE-06-web.jpg": WEB_ROOT / "education-guides" / "GUIDE-06-web.jpg",
    "/education-guides/home-learning-guide-v1.png": WEB_ROOT / "education-guides" / "home-learning-guide-v1.png",
}
LOGGER = logging.getLogger("gene_evidence_agent.web")
SOURCE_NAMES = ("pubmed", "uniprot", "opentargets")
WEB_TOKEN_BUDGET = MODEL_LIFETIME_TOKEN_BUDGET
WEB_P5_TOKEN_BUDGET = MODEL_LOCALIZATION_TOKEN_BUDGET
WEB_P6_E2E_TOKEN_BUDGET = MODEL_P6_E2E_TOKEN_BUDGET
WEB_REPAIR_TOKEN_BUDGET = MODEL_REPAIR_TOKEN_BUDGET
WEB_LOCALIZATION_MAX_REQUEST_TOKENS = WEB_P5_TOKEN_BUDGET
WEB_MODEL_BUDGET_STAGES = {
    "p5": (P5_LOCALIZATION_BUDGET_SCOPE, WEB_P5_TOKEN_BUDGET, "P5 中文整理"),
    "p6-e2e": (P6_E2E_BUDGET_SCOPE, WEB_P6_E2E_TOKEN_BUDGET, "P6 端到端验收"),
    "repair-e2e": (REPAIR_E2E_BUDGET_SCOPE, WEB_REPAIR_TOKEN_BUDGET, "问题修复端到端复测"),
}
# Keep the earlier P6/repair calls in the same persistent scopes. The raised
# gates permit at most three additional attempts per stage without erasing the
# historical request count or token reservations.
WEB_MODEL_BUDGET_REQUEST_LIMITS = {"p6-e2e": 4, "repair-e2e": 19}


class LocalhostHTTPServer(ThreadingHTTPServer):
    """Small threaded server explicitly defaulting to IPv4 loopback only."""

    daemon_threads = True
    allow_reuse_address = False

    def server_close(self):
        try:
            if hasattr(self, "workflow_service"):
                self.workflow_service.close()
            if hasattr(self, "chat_service"):
                self.chat_service.close()
            if hasattr(self, "education_service"):
                self.education_service.close()
        finally:
            super().server_close()


class GeneEvidenceRequestHandler(BaseHTTPRequestHandler):
    """Serve packaged frontend assets and a mock-only JSON endpoint."""

    server_version = "GeneEvidenceLocal/0.1"
    sys_version = ""

    def log_message(self, format_string: str, *args: Any) -> None:
        # Never log request bodies, search terms, headers, or credentials.
        LOGGER.info("client=%s method=%s", self.client_address[0], self.command)

    def _valid_host_and_origin(self) -> bool:
        expected_host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host", "").casefold() != expected_host:
            self._json_response(403, {"error": "invalid_host"})
            return False
        origin = self.headers.get("Origin")
        expected_origin = f"http://127.0.0.1:{self.server.server_port}"
        if origin is not None and origin != expected_origin:
            self._json_response(403, {"error": "invalid_origin"})
            return False
        return True

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data: https://www.cancer.gov; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        if not self._valid_host_and_origin():
            return
        current_path = urlsplit(self.path).path
        if current_path in {"/api/education/catalog", "/api/education/entry"}:
            try:
                params = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                if any(len(value) != 1 for value in params.values()):
                    raise ValueError("duplicate parameter")
                if current_path == "/api/education/catalog":
                    if not set(params).issubset({"kind", "q"}):
                        raise ValueError("unknown parameter")
                    kind = params.get("kind", [None])[0]
                    if kind not in {None, "cancer", "gene", "concept"}:
                        raise ValueError("invalid kind")
                    query = params.get("q", [""])[0]
                    if len(query) > 120:
                        raise ValueError("query too long")
                    entries = self.server.education_service.catalog.search(kind=kind, query=query)
                    self._json_response(200, {
                        "entries": [self.server.education_service.catalog.public(item) for item in entries],
                        "scope_notice": self.server.education_service.catalog.scope_notice(kind=kind, query=query),
                    })
                else:
                    if set(params) != {"id"}:
                        raise ValueError("missing id")
                    entry = self.server.education_service.catalog.get(params["id"][0])
                    if entry is None:
                        self._json_response(404, {"error": "education_entry_not_found"})
                    else:
                        self._json_response(200, self.server.education_service.catalog.public(entry, detail=True))
            except ValueError:
                self._json_response(400, {"error": "invalid_request"})
            return
        if current_path == "/api/education/chat/job":
            try:
                params = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                if set(params) != {"session_id", "job_id"} or any(len(v) != 1 for v in params.values()):
                    raise ValueError("invalid parameters")
                job = self.server.education_service.get_job(UUID(params["session_id"][0]), UUID(params["job_id"][0]))
                self._json_response(200, job.model_dump(mode="json"))
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            except ValueError:
                self._json_response(400, {"error": "invalid_request"})
            return
        if urlsplit(self.path).path in {"/api/chat/job", "/api/synthesis/job", "/api/workflow/state", "/api/workflow/chat/job"}:
            try:
                params = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                current_path = urlsplit(self.path).path
                if current_path == "/api/workflow/state":
                    if set(params) != {"session_id"} or any(len(v) != 1 for v in params.values()):
                        raise ValueError("invalid parameters")
                    state = self.server.workflow_service.state(UUID(params["session_id"][0]))
                    self._json_response(200, state)
                    return
                if current_path == "/api/workflow/chat/job":
                    if set(params) != {"session_id", "job_id"} or any(len(v) != 1 for v in params.values()):
                        raise ValueError("invalid parameters")
                    job = self.server.workflow_service.get_chat_job(UUID(params["session_id"][0]), UUID(params["job_id"][0]))
                    self._json_response(200, job)
                    return
                owner_field = "result_id" if urlsplit(self.path).path == "/api/synthesis/job" else "session_id"
                if set(params) != {owner_field, "job_id"} or any(len(v) != 1 for v in params.values()):
                    raise ValueError("invalid parameters")
                lookup = self.server.chat_service.get_synthesis_job if owner_field == "result_id" else self.server.chat_service.get_job
                job = lookup(UUID(params[owner_field][0]), UUID(params["job_id"][0]))
                self._json_response(200, job.model_dump(mode="json"))
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            except WorkflowError as error:
                self._json_response(error.status, {"error": error.code})
            except ValueError:
                self._json_response(400, {"error": "invalid_request"})
            return
        if self.path == "/api/health":
            budget = self.server.model_budget
            if self.server.model_budget_provider is not None:
                try:
                    budget = self.server.model_budget_provider()
                    self.server.model_budget = budget
                    self.server.model_localization_enabled = bool(budget.get("can_call"))
                    self.server.workflow_service.localization_enabled = False
                except Exception:
                    budget = _unknown_model_budget()
                    self.server.model_budget = budget
                    self.server.model_localization_enabled = False
                    self.server.workflow_service.localization_enabled = False
            self._json_response(
                200,
                {
                    "status": "ok",
                    "instance": self.server.instance_info,
                    "mode": self.server.mode,
                    "model_api_enabled": self.server.model_localization_enabled,
                    "model_budget": budget,
                    "supported_sources": SOURCE_NAMES,
                    "include_drugs_supported": True,
                    "chat": self.server.chat_service.health(),
                    "synthesis": self.server.chat_service.staged_synthesis_engine.health(),
                    "guided_workflow": {
                        "enabled": True,
                        "model_interpretation_enabled": self.server.workflow_service.interpreter is not None,
                        "retrieval_mode": self.server.mode,
                        "localization_enabled": False,
                        "synthesis_enabled": self.server.chat_service.staged_synthesis_engine.health()["enabled"],
                    },
                    "education": self.server.education_service.health(),
                },
            )
            return
        file_path = STATIC_FILES.get(self.path)
        if file_path is None:
            self._json_response(404, {"error": "not_found"})
            return
        try:
            body = file_path.read_bytes()
        except OSError:
            self._json_response(500, {"error": "asset_unavailable"})
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        content_type_header = (
            f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type
        )
        self.send_header("Content-Type", content_type_header)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        if not self._valid_host_and_origin():
            return
        education_paths = {"/api/education/chat/session", "/api/education/chat/message", "/api/education/chat/clear"}
        chat_paths = {"/api/chat/session", "/api/chat/message", "/api/chat/clear"}
        workflow_paths = {
            "/api/workflow/session", "/api/workflow/message", "/api/workflow/prepare",
            "/api/workflow/confirm", "/api/workflow/select-result",
            "/api/workflow/resolve-entity",
            "/api/workflow/synthesis-bind",
        }
        synthesis_paths = {"/api/synthesis", "/api/synthesis/action"}
        allowed_paths = {"/api/query", "/api/attachments"} | synthesis_paths | chat_paths | workflow_paths | education_paths
        if self.path not in allowed_paths:
            self._json_response(404, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json_response(415, {"error": "json_required"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json_response(400, {"error": "invalid_content_length"})
            return
        body_limit = 32768 if self.path in chat_paths | workflow_paths | education_paths | {"/api/synthesis/action"} else MAX_BODY_BYTES
        if self.path == "/api/attachments":
            body_limit = MAX_UPLOAD_BODY
        if not 1 <= content_length <= body_limit:
            self._json_response(413 if content_length > body_limit else 400, {"error": "invalid_body_size"})
            return
        try:
            request_data = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json_response(400, {"error": "invalid_json"})
            return
        if self.path == "/api/attachments":
            try:
                request = UploadRequest.model_validate(request_data)
                service = self.server.chat_service
                owner = request.session_id
                if request.channel == "education":
                    education = self.server.education_service
                    with education._lock:
                        if owner not in education._sessions:
                            raise ChatServiceError("session_not_found", 404)
                    service = education.chat_service
                elif request.channel == "workflow":
                    workflow = self.server.workflow_service
                    with workflow.lock:
                        owner = workflow._session(owner).chat_session_id
                result = service.upload_attachment(owner, request.filename, request.content_base64)
                self._json_response(201, result)
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except (ChatServiceError, WorkflowError) as error:
                self._json_response(error.status, {"error": error.code})
            return
        if self.path in education_paths:
            try:
                if self.path == "/api/education/chat/session":
                    request = EducationSessionRequest.model_validate(request_data)
                    self._json_response(201, self.server.education_service.create_session(request.entry_id))
                elif self.path == "/api/education/chat/message":
                    request = EducationMessageRequest.model_validate(request_data)
                    self._json_response(202, self.server.education_service.submit(request).model_dump(mode="json"))
                else:
                    request = EducationClearRequest.model_validate(request_data)
                    self._json_response(200, self.server.education_service.clear(request.session_id))
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            return
        if self.path == "/api/synthesis":
            try:
                request = SynthesisRequest.model_validate(request_data)
                self._json_response(202, self.server.chat_service.start_staged_synthesis(request.result_id).model_dump(mode="json"))
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            return
        if self.path == "/api/synthesis/action":
            try:
                request = SynthesisActionRequest.model_validate(request_data)
                job = self.server.chat_service.confirm_staged_synthesis(request)
                self._json_response(202, job.model_dump(mode="json"))
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            return
        if self.path in workflow_paths:
            if self.path == "/api/workflow/prepare" and isinstance(request_data, dict) and request_data.get("action") == "localize":
                self._json_response(410, {"error": "localization_removed"})
                return
            try:
                if self.path == "/api/workflow/session":
                    WorkflowSessionRequest.model_validate(request_data)
                    self._json_response(201, self.server.workflow_service.create_session())
                elif self.path == "/api/workflow/message":
                    request = WorkflowMessageRequest.model_validate(request_data)
                    self._json_response(202, self.server.workflow_service.message(request))
                elif self.path == "/api/workflow/prepare":
                    request = WorkflowPrepareActionRequest.model_validate(request_data)
                    self._json_response(200, self.server.workflow_service.prepare_action(request))
                elif self.path == "/api/workflow/confirm":
                    request = WorkflowConfirmActionRequest.model_validate(request_data)
                    self._json_response(202, self.server.workflow_service.confirm_action(request))
                elif self.path == "/api/workflow/synthesis-bind":
                    request = WorkflowBindSynthesisRequest.model_validate(request_data)
                    self._json_response(200, self.server.workflow_service.bind_synthesis(request))
                else:
                    if self.path == "/api/workflow/select-result":
                        request = WorkflowSelectResultRequest.model_validate(request_data)
                        self._json_response(200, self.server.workflow_service.select_result(request))
                    else:
                        request = WorkflowResolveEntityRequest.model_validate(request_data)
                        self._json_response(200, self.server.workflow_service.resolve_entity(request))
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except WorkflowError as error:
                self._json_response(error.status, {"error": error.code})
            return
        if self.path in chat_paths:
            try:
                if self.path == "/api/chat/session":
                    ChatSessionRequest.model_validate(request_data)
                    self._json_response(201, self.server.chat_service.create_session())
                elif self.path == "/api/chat/clear":
                    request = ChatClearRequest.model_validate(request_data)
                    self._json_response(200, self.server.chat_service.clear(request.session_id))
                else:
                    request = ChatMessageRequest.model_validate(request_data)
                    self._json_response(202, self.server.chat_service.submit(request).model_dump(mode="json"))
            except ValidationError:
                self._json_response(400, {"error": "invalid_request"})
            except ChatServiceError as error:
                self._json_response(error.status, {"error": error.code})
            return
        allowed_fields = {"gene", "disease", "sources", "include_drugs", "localize_zh"}
        if not isinstance(request_data, dict) or not {"gene", "disease"}.issubset(request_data) or not set(request_data).issubset(allowed_fields):
            self._json_response(400, {"error": "expected_supported_query_fields"})
            return
        gene = _validate_query_field(request_data.get("gene"))
        disease = _validate_query_field(request_data.get("disease"))
        if gene is None or disease is None:
            self._json_response(400, {"error": "invalid_query_field"})
            return
        if self.server.mode == "live":
            if contains_chinese_text(gene):
                self._json_response(400, {"error": "english_gene_symbol_required"})
                return
            try:
                normalize_disease_search_term(disease)
            except UnsupportedChineseDiseaseTerm:
                self._json_response(400, {"error": "unsupported_chinese_disease_term"})
                return
        sources_value = request_data.get("sources", list(SOURCE_NAMES))
        if (
            not isinstance(sources_value, list)
            or not sources_value
            or len(sources_value) > len(SOURCE_NAMES)
            or any(source not in SOURCE_NAMES for source in sources_value)
            or len(set(sources_value)) != len(sources_value)
        ):
            self._json_response(400, {"error": "invalid_sources"})
            return
        include_drugs = request_data.get("include_drugs", True)
        if not isinstance(include_drugs, bool):
            self._json_response(400, {"error": "include_drugs_must_be_boolean"})
            return
        localize_zh = request_data.get("localize_zh", False)
        if not isinstance(localize_zh, bool):
            self._json_response(400, {"error": "localize_zh_must_be_boolean"})
            return
        if localize_zh:
            self._json_response(410, {"error": "localization_removed"})
            return
        if include_drugs and "opentargets" not in sources_value:
            self._json_response(400, {"error": "drug_option_requires_opentargets"})
            return
        try:
            ensure_no_sensitive_query(f"gene {gene}; disease {disease}")
        except InvalidInputError:
            self._json_response(400, {"error": "sensitive_query_rejected"})
            return
        try:
            result = self.server.query_handler(
                gene, disease, set(sources_value), include_drugs
            )
        except Exception as error:
            LOGGER.error("web query failed; type=%s", type(error).__name__)
            self._json_response(502, {"error": "query_failed"})
            return
        result["result_id"] = self.server.chat_service.register_report(result)
        self._json_response(200, result)


def _validate_query_field(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 120 or any(ord(char) < 32 for char in cleaned):
        return None
    return cleaned


def _mock_result(
    gene: str,
    disease: str,
    sources: set[str] | None = None,
    include_drugs: bool = True,
) -> dict[str, Any]:
    """Create explicitly synthetic UI data; this does not resemble retrieved evidence."""

    retrieved_at = datetime.now(timezone.utc).isoformat()
    demo_entities = [
        {"id": "demo:gene", "entity_type": "gene", "query_name": gene, "canonical_name": None, "identifier": None, "resolution_status": "unresolved"},
        {"id": "demo:disease", "entity_type": "disease", "query_name": disease, "canonical_name": None, "identifier": None, "resolution_status": "unresolved"},
    ]
    demo_records = [
        ("publication", "MOCK-PUBMED-001", "模拟论文标题占位（并非真实论文）", "pubmed"),
        ("protein_annotation", "MOCK-UNIPROT-001", "模拟蛋白注释占位", "uniprot"),
        ("protein_annotation", "MOCK-UNIPROT-002", "第二条模拟蛋白注释占位", "uniprot"),
        ("target_disease_association", "MOCK-OT-ASSOCIATION-001", "模拟靶点—疾病关联占位", "opentargets"),
        ("disease_drug_candidate", "MOCK-OT-DRUG-001", "模拟疾病层级药物候选占位", "opentargets"),
    ]
    demo_records = [
        record for record in demo_records
        if (sources is None or record[3] in sources)
        and (include_drugs or record[0] != "disease_drug_candidate")
    ]
    counts = {source: sum(record[3] == source for record in demo_records) for source in SOURCE_NAMES}
    return {
        "mode": "mock",
        "status": "partial_success",
        "query": f"Gene {gene}; disease {disease}",
        "summary": "这是本机界面联调模拟信息，不是对基因、疾病或治疗效果的真实判断；当前没有真实来源记录。",
        "source_outcomes": [
            {
                "source": source,
                "status": "not_requested" if counts[source] == 0 else "ambiguous" if source == "uniprot" else "success",
                "record_count": counts[source],
            }
            for source in SOURCE_NAMES
        ],
        "entities": demo_entities,
        "evidence": [
            {
                "evidence_type": evidence_type,
                "source": {
                    "source_database": "模拟数据（无真实来源）",
                    "record_id": record_id,
                    "record_url": None,
                    "retrieved_at": retrieved_at,
                },
                "entity_ids": ["demo:gene", "demo:disease"],
                "title": title,
                "summary": "仅用于展示网页卡片和来源字段布局的虚构占位文本；不包含真实生物医学信息。",
            }
            for evidence_type, record_id, title, _source in demo_records
        ],
        "warnings": ["mock_response_no_external_sources_or_model_called"],
    }


class LivePipelineQueryHandler:
    """Opt-in deterministic source adapter; it never creates a model client."""

    def __init__(
        self,
        settings: Settings,
        *,
        pipeline: Any | None = None,
        adapter_factory: Any | None = None,
        model_budget_stage: str = "p5",
    ) -> None:
        if model_budget_stage not in WEB_MODEL_BUDGET_STAGES:
            raise ValueError("Unknown web model budget stage")
        self._pipeline = pipeline or GeneEvidencePipeline(settings)
        self._settings = settings
        self._adapter_factory = adapter_factory
        self._model_budget_stage = model_budget_stage

    def __call__(
        self,
        gene_name: str,
        disease_name: str,
        sources: set[str],
        include_drugs: bool,
        *,
        localize_zh: bool = False,
    ) -> dict[str, Any]:
        return asyncio.run(
            self._collect(gene_name, disease_name, sources, include_drugs, localize_zh)
        )

    async def _collect(
        self,
        gene_name: str,
        disease_name: str,
        sources: set[str],
        include_drugs: bool,
        localize_zh: bool,
    ) -> dict[str, Any]:
        gene = _query_entity("gene", gene_name)
        disease = _query_entity("disease", disease_name)
        collection = await self._pipeline.collect(
            gene,
            disease,
            limit=self._settings.max_results_per_source,
            sources=sources,
            include_drugs=include_drugs,
        )
        reported_disease_name = collection.disease.canonical_name
        query = f"Gene {gene_name}; disease {disease_name}"
        if reported_disease_name and reported_disease_name != disease_name:
            query += f" (English search term: {reported_disease_name})"
        report = collection.to_evidence_report(query=query)
        payload = report.model_dump(mode="json", exclude_none=True)
        from gene_evidence_agent.chinese_report import stable_evidence_id

        for record, record_payload in zip(report.evidence, payload["evidence"], strict=True):
            record_payload["evidence_id"] = stable_evidence_id(record)
        result = {
            "mode": "live",
            "status": "partial_success" if any(
                outcome.status not in {"success", "empty", "not_requested"}
                for outcome in report.source_outcomes
            ) else "completed",
            **payload,
        }
        if localize_zh:
            from gene_evidence_agent.chinese_report import localize_report, TRANSLATION_INPUT_RESERVE_MARGIN

            adapter = None
            if self._settings.model_enabled:
                if self._adapter_factory is None:
                    from gene_evidence_agent.model_adapter import OpenAICompatibleModelAdapter

                    adapter_factory = OpenAICompatibleModelAdapter
                else:
                    adapter_factory = self._adapter_factory
                adapter = adapter_factory(
                    self._settings,
                    max_lifetime_tokens=WEB_TOKEN_BUDGET,
                    allowed_uncertain_requests=self._settings.model_allowed_uncertain_requests,
                    budget_scope=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][0],
                    max_scope_tokens=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][1],
                    max_request_tokens=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][1],
                    max_scope_requests=WEB_MODEL_BUDGET_REQUEST_LIMITS.get(self._model_budget_stage),
                    input_token_reserve_margin=TRANSLATION_INPUT_RESERVE_MARGIN,
                )
            try:
                chinese_report = await localize_report(
                    report,
                    settings=self._settings,
                    adapter=adapter,
                    timeout_seconds=self._settings.agent_run_timeout_seconds,
                )
            finally:
                if adapter is not None:
                    await adapter.aclose()
            result["chinese_report"] = chinese_report.model_dump(
                mode="json", exclude_none=True
            )
        return result

    def localize_existing(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Localize a previously retrieved report without repeating source queries."""
        from gene_evidence_agent.chinese_report import localize_report
        from gene_evidence_agent.schemas import EvidenceRecord, EvidenceReport

        report_payload = {key: value for key, value in payload.items() if key in EvidenceReport.model_fields}
        report_payload["evidence"] = [
            {key: value for key, value in record.items() if key in EvidenceRecord.model_fields}
            for record in report_payload.get("evidence", [])
        ]
        report = EvidenceReport.model_validate(report_payload)
        return asyncio.run(self._localize_existing(report))

    async def _localize_existing(self, report):
        from gene_evidence_agent.chinese_report import TRANSLATION_INPUT_RESERVE_MARGIN, localize_report

        adapter = None
        if self._settings.model_enabled:
            if self._adapter_factory is None:
                from gene_evidence_agent.model_adapter import OpenAICompatibleModelAdapter
                adapter_factory = OpenAICompatibleModelAdapter
            else:
                adapter_factory = self._adapter_factory
            adapter = adapter_factory(
                self._settings,
                max_lifetime_tokens=WEB_TOKEN_BUDGET,
                allowed_uncertain_requests=self._settings.model_allowed_uncertain_requests,
                budget_scope=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][0],
                max_scope_tokens=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][1],
                max_request_tokens=WEB_MODEL_BUDGET_STAGES[self._model_budget_stage][1],
                max_scope_requests=WEB_MODEL_BUDGET_REQUEST_LIMITS.get(self._model_budget_stage),
                input_token_reserve_margin=TRANSLATION_INPUT_RESERVE_MARGIN,
            )
        try:
            localized = await localize_report(
                report, settings=self._settings, adapter=adapter,
                timeout_seconds=self._settings.agent_run_timeout_seconds,
            )
            return localized.model_dump(mode="json", exclude_none=True)
        finally:
            if adapter is not None:
                await adapter.aclose()


def _query_entity(entity_type: str, query_name: str) -> BiomedicalEntity:
    digest = hashlib.sha256(query_name.casefold().encode("utf-8")).hexdigest()[:20]
    return BiomedicalEntity(
        id=f"query:{entity_type}:{digest}",
        entity_type=entity_type,  # validated below by Pydantic
        query_name=query_name,
    )


def create_server(
    host: str = HOST,
    port: int = PORT,
    *,
    mode: str = "mock",
    query_handler: Any | None = None,
    model_localization_enabled: bool = False,
    model_budget: dict[str, Any] | None = None,
    model_budget_provider: Any | None = None,
    chat_service: ResearchChatService | None = None,
    workflow_service: GuidedWorkflowService | None = None,
    workflow_interpreter: Any | None = None,
    localization_enabled: bool = False,
    education_service: EducationChat | None = None,
) -> LocalhostHTTPServer:
    """Create the server; callers should keep the host at the loopback default."""

    if host != HOST:
        raise ValueError("Web server may only bind to 127.0.0.1")
    if mode not in {"mock", "live"}:
        raise ValueError("Web mode must be 'mock' or 'live'")
    server = LocalhostHTTPServer((host, port), GeneEvidenceRequestHandler)
    server.instance_info = {
        "application": "gene-evidence-agent",
        "application_version": __version__,
        "instance_id": str(uuid4()),
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    server.mode = mode
    server.query_handler = query_handler or _mock_result
    server.model_localization_enabled = bool(model_localization_enabled)
    server.model_budget = model_budget or {
        "token_limit": WEB_TOKEN_BUDGET,
        "p5_token_limit": WEB_P5_TOKEN_BUDGET,
        "stage": "p5",
        "stage_label": WEB_MODEL_BUDGET_STAGES["p5"][2],
        "stage_token_limit": WEB_P5_TOKEN_BUDGET,
        "used_tokens": None,
        "remaining_tokens": None,
        "p5_used_tokens": None,
        "p5_remaining_tokens": None,
        "stage_used_tokens": None,
        "stage_remaining_tokens": None,
        "stage_request_count": None,
        "stage_request_limit": None,
        "uncertain_requests": 0,
        "can_call": False,
        "status": "disabled",
    }
    server.model_budget_provider = model_budget_provider
    server.chat_service = chat_service or ResearchChatService(mode=mode)
    server.workflow_service = workflow_service or GuidedWorkflowService(
        mode=mode,
        chat_service=server.chat_service,
        query_handler=server.query_handler,
        interpreter=workflow_interpreter,
        localization_handler=None,
        localization_enabled=False,
    )
    if education_service is None:
        catalog = EducationCatalog()
        education_chat = ResearchChatService(mode=mode, knowledge=catalog.knowledge)
        education_service = EducationChat(catalog, education_chat)
    server.education_service = education_service
    return server


def _model_budget_status(
    settings: Settings, *, model_opt_in: bool, budget_stage: str = "p5"
) -> dict[str, Any]:
    """Return sanitized current web token budget state; uncertainty fails closed."""

    if budget_stage not in WEB_MODEL_BUDGET_STAGES:
        return _unknown_model_budget()
    budget_scope, stage_token_limit, stage_label = WEB_MODEL_BUDGET_STAGES[budget_stage]
    ledger = ModelUsageLedger(settings.model_usage_db_path)
    lifetime = ledger.token_budget_snapshot()
    p5 = ledger.token_budget_snapshot(budget_scope=P5_LOCALIZATION_BUDGET_SCOPE)
    stage_usage = ledger.token_budget_snapshot(budget_scope=budget_scope)
    remaining = max(0, WEB_TOKEN_BUDGET - lifetime.used_tokens)
    p5_remaining = max(0, WEB_P5_TOKEN_BUDGET - p5.used_tokens)
    stage_remaining = max(0, stage_token_limit - stage_usage.used_tokens)
    credentials_ready = bool(
        model_opt_in
        and settings.model_enabled
        and settings.model_api_key
        and settings.model_name
        and settings.model_base_url
    )
    if settings.model_freeze_on_uncertain_usage and lifetime.uncertain_requests > settings.model_allowed_uncertain_requests:
        status = "uncertain_usage"
    elif remaining == 0:
        status = "budget_exhausted"
    elif (
        budget_stage in WEB_MODEL_BUDGET_REQUEST_LIMITS
        and stage_usage.request_count >= WEB_MODEL_BUDGET_REQUEST_LIMITS[budget_stage]
    ):
        status = "stage_request_limit_reached"
    elif stage_remaining == 0:
        status = "stage_budget_exhausted"
    elif stage_remaining <= settings.model_max_output_tokens:
        status = "stage_budget_insufficient"
    elif not credentials_ready:
        status = "disabled_or_unconfigured"
    elif lifetime.uncertain_requests:
        status = "available_with_historical_uncertainty"
    else:
        status = "available"
    can_call = status in {"available", "available_with_historical_uncertainty"}
    return {
        "token_limit": WEB_TOKEN_BUDGET,
        "p5_token_limit": WEB_P5_TOKEN_BUDGET,
        "stage": budget_stage,
        "stage_label": stage_label,
        "stage_token_limit": stage_token_limit,
        "used_tokens": lifetime.used_tokens,
        "remaining_tokens": remaining,
        "p5_used_tokens": p5.used_tokens,
        "p5_remaining_tokens": p5_remaining,
        "stage_used_tokens": stage_usage.used_tokens,
        "stage_remaining_tokens": stage_remaining,
        "stage_request_count": stage_usage.request_count,
        "stage_request_limit": WEB_MODEL_BUDGET_REQUEST_LIMITS.get(budget_stage),
        "uncertain_requests": lifetime.uncertain_requests,
        "freeze_on_uncertain_usage": settings.model_freeze_on_uncertain_usage,
        "can_call": can_call,
        "status": status,
    }


def _unknown_model_budget() -> dict[str, Any]:
    return {
        "token_limit": WEB_TOKEN_BUDGET,
        "p5_token_limit": WEB_P5_TOKEN_BUDGET,
        "stage": "unknown",
        "stage_label": "当前阶段",
        "stage_token_limit": None,
        "used_tokens": None,
        "remaining_tokens": None,
        "p5_used_tokens": None,
        "p5_remaining_tokens": None,
        "stage_used_tokens": None,
        "stage_remaining_tokens": None,
        "stage_request_count": None,
        "stage_request_limit": None,
        "uncertain_requests": None,
        "can_call": False,
        "status": "unknown",
    }


def main() -> None:
    mode = os.environ.get("GEA_WEB_MODE", "mock").strip().casefold()
    try:
        port = int(os.environ.get("GEA_WEB_PORT", str(PORT)))
    except ValueError as error:
        raise ValueError("GEA_WEB_PORT must be an integer port") from error
    if not 0 <= port <= 65535:
        raise ValueError("GEA_WEB_PORT must be between 0 and 65535")
    handler = None
    model_localization_enabled = False
    model_budget: dict[str, Any] | None = None
    model_budget_provider = None
    log_level: str = "INFO"
    chat_service = None
    education_service = None
    workflow_interpreter = None
    if mode == "live":
        # A separate process-environment opt-in is required; the browser cannot enable model calls.
        model_opt_in = os.environ.get("GEA_WEB_MODEL_ENABLED", "").strip().casefold() == "true"
        budget_stage = os.environ.get("GEA_WEB_MODEL_BUDGET_STAGE", "p5").strip().casefold()
        if budget_stage not in WEB_MODEL_BUDGET_STAGES:
            raise ValueError("GEA_WEB_MODEL_BUDGET_STAGE must be 'p5', 'p6-e2e', or 'repair-e2e'")
        if model_opt_in:
            settings = Settings(
                model_enabled=True,
                chinese_claims_mode=os.environ.get("GEA_CHINESE_CLAIMS_MODE", "extractive").strip().casefold(),
            )
        else:
            # Keep model configuration out of the active settings when the explicit opt-in is absent.
            settings = Settings(
                model_enabled=False,
                model_api_key=None,
                model_base_url=None,
                model_name=None,
            )
        log_level = settings.log_level
        handler = LivePipelineQueryHandler(settings, model_budget_stage=budget_stage)
        model_budget_provider = lambda: _model_budget_status(
            settings, model_opt_in=model_opt_in, budget_stage=budget_stage
        )
        try:
            model_budget = model_budget_provider()
            model_localization_enabled = bool(model_budget["can_call"])
        except Exception:
            # Unknown budget state is fail-closed; the adapter rechecks before network I/O as well.
            model_budget = _unknown_model_budget()
        if os.environ.get("GEA_WEB_CHAT_ENABLED", "").strip().casefold() == "true":
            from gene_evidence_agent.chat_model import ChatModelResponder
            chat_settings = Settings(model_enabled=True)
            validation = os.environ.get("GEA_CHAT_VALIDATION_MODE", "").strip().casefold() == "true"
            responder = ChatModelResponder(chat_settings, max_scope_tokens=100000 if validation else 10000000,
                                           max_scope_requests=3 if validation else None)
            chat_service = ResearchChatService(mode="live", responder=responder)
            if model_opt_in:
                workflow_interpreter = GuidedWorkflowInterpreter(chat_settings)
        education_catalog = EducationCatalog()
        education_responder = None
        if (
            model_opt_in
            and os.environ.get("GEA_WEB_EDUCATION_CHAT_ENABLED", "").strip().casefold() == "true"
        ):
            from gene_evidence_agent.chat_model import ChatModelResponder
            education_settings = Settings(model_enabled=True)
            education_responder = ChatModelResponder(
                education_settings,
                max_scope_tokens=EDUCATION_STAGE_CAP,
                budget_scope=EDUCATION_SCOPE,
                output_cap=EDUCATION_OUTPUT_CAP,
                system_prompt=EDUCATION_SYSTEM_PROMPT,
            )
        education_chat = ResearchChatService(
            mode="live", responder=education_responder, knowledge=education_catalog.knowledge,
        )
        education_service = EducationChat(education_catalog, education_chat)
        if os.environ.get("GEA_WEB_SYNTHESIS_ENABLED", "").strip().casefold() == "true":
            from gene_evidence_agent.chat_model import ChatModelResponder
            synthesis_settings = Settings(model_enabled=True)
            validation = os.environ.get("GEA_SYNTHESIS_VALIDATION_MODE", "").strip().casefold() == "true"
            # New six-request acceptance cycle starts after the seven historical
            # requests. Keep the same ledger/scope; never delete usage history.
            scope_cap, request_cap = (2000000, 13) if validation else (10000000, None)
            engine = SynthesisEngine(synthesis_settings, max_scope_tokens=scope_cap, max_scope_requests=request_cap, stop_on_failure=validation)
            staged_engine = EnglishSynthesisEngine(
                synthesis_settings, max_scope_tokens=scope_cap,
                max_scope_requests=request_cap, stop_on_failure=validation,
            )
            if chat_service is None:
                chat_service = ResearchChatService(mode="live")
            chat_service.synthesis_engine = engine
            chat_service.staged_synthesis_engine = staged_engine
            if os.environ.get("GEA_WEB_CHAT_ENABLED", "").strip().casefold() == "true":
                chat_service.synthesis_responder = ChatModelResponder(synthesis_settings, max_scope_tokens=scope_cap,
                    max_scope_requests=request_cap, budget_scope=SYNTHESIS_SCOPE)
    # Reuse the query-redacting filter so HTTPX URLs never leak PubMed contact
    # parameters or future API credentials to the terminal log.
    configure_logging(log_level)
    server = create_server(
        port=port,
        mode=mode,
        query_handler=handler,
        model_localization_enabled=model_localization_enabled,
        model_budget=model_budget,
        model_budget_provider=model_budget_provider,
        chat_service=chat_service,
        workflow_interpreter=workflow_interpreter,
        localization_enabled=model_localization_enabled,
        education_service=education_service,
    )
    print(f"Gene Evidence web UI ({mode}): http://127.0.0.1:{server.server_port}/")
    if mode == "mock":
        print("网页服务仅监听 127.0.0.1；默认模拟模式不访问模型 API 或外部数据库。按 Ctrl+C 停止。")
    else:
        print("网页服务仅监听 127.0.0.1；查询词将发送给所选外部数据库。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在关闭本机网页服务…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
