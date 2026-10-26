"""Small, source-linked cancer and gene education catalog with isolated chat."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, HttpUrl

from gene_evidence_agent.chat import ResearchChatService, ChatServiceError
from gene_evidence_agent.chat_schemas import ChatCitation, ChatMessageRequest
from gene_evidence_agent.errors import InvalidInputError
from gene_evidence_agent.knowledge_store import KnowledgeCard, KnowledgeExcerpt, KnowledgeStore
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.schemas import StrictModel

EDUCATION_ROOT = Path(__file__).with_name("education")
EDUCATION_CATALOG = EDUCATION_ROOT / "catalog.json"
EDUCATION_SCOPE = "education-chat"
EDUCATION_OUTPUT_CAP = 1800
EDUCATION_STAGE_CAP = 1_000_000
EDUCATION_CHAT_CONTEXT_TOPIC_LIMIT = 3
EDUCATION_IMPORTED_CARD_LIMIT = 6
_GENERIC_TOPIC_TERMS = {"癌", "癌症", "肿瘤", "cancer", "cancers", "tumor", "tumour", "基因", "gene", "genes"}
_GENERIC_TOPIC_PHRASES = ("是什么", "什么是", "如何", "怎么", "含义", "意思", "基础知识", "概况", "简介", "介绍")

_EDUCATION_SCOPE_NOTICES = (
    {
        "kind": "cancer",
        "terms": ("uterine cancer", "子宫癌"),
        "specific_terms": (
            "endometrial", "子宫内膜", "uterine sarcoma", "子宫肉瘤",
            "cervical cancer", "宫颈癌", "cervix",
        ),
        "title": "先确认子宫癌的具体范围",
        "message": (
            "“子宫癌”是较宽泛的说法。NCI 的子宫癌专题分别介绍子宫内膜癌和子宫肉瘤；"
            "目录也将宫颈癌作为独立主题。先看下方分类入口，再按你想了解的具体类型选择词条。"
        ),
        "sources": (
            {"title": "NCI：Uterine Cancer", "url": "https://www.cancer.gov/types/uterine"},
            {"title": "NCI：What Is Uterine Sarcoma?", "url": "https://www.cancer.gov/types/uterine/sarcoma"},
            {"title": "NCI：What Is Cervical Cancer?", "url": "https://www.cancer.gov/types/cervical"},
            {
                "title": "ACS：What Is Endometrial Cancer?",
                "url": "https://www.cancer.org/cancer/types/endometrial-cancer/about/what-is-endometrial-cancer.html",
            },
            {
                "title": "ACS：Uterine Sarcoma",
                "url": "https://www.cancer.org/cancer/types/uterine-sarcoma.html",
            },
        ),
    },
    {
        "kind": "cancer",
        "terms": ("melanoma", "黑色素瘤"),
        "specific_terms": ("cutaneous melanoma", "skin melanoma", "melanoma of the skin", "皮肤黑色素瘤"),
        "title": "先确认黑色素瘤的发生部位",
        "message": (
            "“黑色素瘤”是较宽泛的名称，可发生在皮肤或其他含色素组织。当前目录提供分类入口和"
            "皮肤黑色素瘤知识卡；眼内或黏膜等其他部位尚无独立词条。请先确认研究或学习对象的具体部位。"
        ),
        "sources": (
            {
                "title": "NCI Dictionary：Melanoma",
                "url": "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/melanoma",
            },
            {
                "title": "NCI：Skin Cancer (Including Melanoma)",
                "url": "https://www.cancer.gov/types/skin",
            },
            {
                "title": "ACS：Melanoma Skin Cancer",
                "url": "https://www.cancer.org/cancer/types/melanoma-skin-cancer.html",
            },
        ),
    },
)


class EducationSource(StrictModel):
    id: str = Field(pattern=r"^SRC-[A-Za-z0-9-]{1,70}$")
    title: str = Field(min_length=1, max_length=300)
    publisher: str = Field(min_length=1, max_length=120)
    url: HttpUrl
    checked_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    review_status: str = Field(pattern=r"^(source_checked|draft)$")


class EducationFaq(StrictModel):
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=1500)
    source_ids: list[str] = Field(min_length=1, max_length=4)


class EducationEntry(StrictModel):
    id: str = Field(pattern=r"^K-edu-[a-z0-9-]{1,60}$")
    kind: str = Field(pattern=r"^(cancer|gene|concept)$")
    category: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=160)
    aliases: list[str] = Field(min_length=1, max_length=20)
    summary: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=3000)
    source_url: HttpUrl
    source_title: str = Field(min_length=1, max_length=300)
    source_date: str | None = Field(default=None, max_length=40)
    checked_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    review_status: str = Field(pattern=r"^(source_checked|draft)$")
    media_ids: list[str] = Field(default_factory=list, max_length=10)
    faqs: list[EducationFaq] = Field(default_factory=list, max_length=4)

    @property
    def keywords(self) -> list[str]:
        return list(dict.fromkeys([self.id, self.name, *self.aliases]))[:30]


class EducationMedia(StrictModel):
    id: str = Field(pattern=r"^MED-[A-Z0-9-]{1,80}$")
    kind: Literal["image", "video"]
    topic_ids: list[str] = Field(min_length=1, max_length=20)
    title: str = Field(min_length=1, max_length=200)
    publisher: str = Field(min_length=1, max_length=120)
    source_page_url: HttpUrl
    source_page_title: str = Field(min_length=1, max_length=200)
    asset_url: str | None = Field(default=None, max_length=1000)
    display_mode: Literal["local_asset", "approved_embed", "external_link"]
    license_status: Literal["pending_review", "rights_checked", "blocked"]
    license_evidence_url: HttpUrl | None = None
    language: str = Field(min_length=2, max_length=40)
    duration_seconds: int | None = Field(default=None, ge=0)
    caption: str = Field(min_length=1, max_length=1200)
    alt_text: str = Field(min_length=1, max_length=1200)
    transcript_status: Literal["available", "not_available", "not_applicable", "unknown"]
    checked_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    publication_status: Literal["draft", "published"]


class EducationMessageRequest(StrictModel):
    session_id: UUID
    message_id: UUID
    entry_id: str = Field(pattern=r"^K-edu-[a-z0-9-]{1,60}$")
    text: str = Field(min_length=1, max_length=1200)
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=4)
    context_entry_ids: list[str] | None = Field(default=None, max_length=EDUCATION_IMPORTED_CARD_LIMIT)


class EducationSessionRequest(StrictModel):
    entry_id: str = Field(pattern=r"^K-edu-[a-z0-9-]{1,60}$")


class EducationClearRequest(StrictModel):
    session_id: UUID


class EducationCatalog:
    def __init__(self, path: Path = EDUCATION_CATALOG):
        self._catalog_path = Path(path)
        catalog_source = self._catalog_path.read_bytes()
        self.loaded_sha256 = hashlib.sha256(catalog_source).hexdigest()
        raw = json.loads(catalog_source.decode("utf-8"))
        if raw.get("schema_version") != 1 or not isinstance(raw.get("entries"), list):
            raise ValueError("invalid education catalog schema")
        raw_media = raw.get("media", [])
        if not isinstance(raw_media, list):
            raise ValueError("invalid education media catalog")
        raw_sources = raw.get("sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError("invalid education source catalog")
        self.entries: dict[str, EducationEntry] = {}
        self.media: dict[str, EducationMedia] = {}
        self.sources: dict[str, EducationSource] = {}
        for item in raw_sources:
            source = EducationSource.model_validate(item)
            if source.id in self.sources or source.url.scheme != "https":
                raise ValueError("invalid or duplicate education source")
            self.sources[source.id] = source
        cards: list[KnowledgeCard] = []
        seen_names: set[tuple[str, str]] = set()
        for item in raw["entries"]:
            entry = EducationEntry.model_validate(item)
            if entry.id in self.entries or (entry.kind, entry.name.casefold()) in seen_names:
                raise ValueError("duplicate education entry")
            for faq in entry.faqs:
                if any(
                    source_id not in self.sources
                    or self.sources[source_id].review_status != "source_checked"
                    for source_id in faq.source_ids
                ):
                    raise ValueError("education FAQ requires checked source records")
            ChatCitation(source_id=entry.id, kind="knowledge", title=entry.name, url=entry.source_url)
            seen_names.add((entry.kind, entry.name.casefold()))
            self.entries[entry.id] = entry
            cards.append(KnowledgeCard(
                id=entry.id,
                title=entry.name,
                keywords=entry.keywords,
                source_url=str(entry.source_url),
                source_title=entry.source_title,
                source_date=entry.source_date,
                checked_at=entry.checked_at,
                review_status=entry.review_status,
                scope="入门科普；来源核对，不代表独立专家审核",
                body_file="education-entry.md",
                body=entry.summary + "\n\n" + entry.body,
                excerpts=[KnowledgeExcerpt(
                    citation=ChatCitation(
                        source_id=f"{entry.id}-faq-{faq_index + 1}-src-{source_index + 1}",
                        kind="knowledge", title=(faq.question + " · " + self.sources[source_id].title)[:500],
                        url=self.sources[source_id].url,
                    ), text=faq.question + "\n" + faq.answer,
                ) for faq_index, faq in enumerate(entry.faqs)
                  for source_index, source_id in enumerate(faq.source_ids)],
            ))
        for item in raw_media:
            media = EducationMedia.model_validate(item)
            if media.id in self.media:
                raise ValueError("duplicate education media id")
            unknown_topics = set(media.topic_ids) - set(self.entries)
            if unknown_topics:
                raise ValueError("education media references unknown topic")
            if media.display_mode == "approved_embed":
                parsed_asset = urlsplit(media.asset_url or "")
                if (
                    media.license_status != "rights_checked"
                    or parsed_asset.scheme != "https"
                    or parsed_asset.hostname not in {"cancer.gov", "www.cancer.gov"}
                    or parsed_asset.username is not None
                    or parsed_asset.password is not None
                    or parsed_asset.port not in {None, 443}
                ):
                    raise ValueError("unapproved external education media")
            if media.publication_status == "published" and (
                media.license_status != "rights_checked" or media.license_evidence_url is None
            ):
                raise ValueError("published education media requires rights evidence")
            self.media[media.id] = media
        for entry in self.entries.values():
            if any(media_id not in self.media for media_id in entry.media_ids):
                raise ValueError("education entry references unknown media")
            if any(entry.id not in self.media[media_id].topic_ids for media_id in entry.media_ids):
                raise ValueError("education media topic binding mismatch")
        self._knowledge_root = path.parent / "empty"
        self.knowledge = KnowledgeStore(root=self._knowledge_root)
        self.knowledge.cards = cards
        catalog_bytes = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.knowledge.version = f"education-v1-{hashlib.sha256(catalog_bytes).hexdigest()[:48]}"
        self._knowledge_cards = {card.id: card for card in cards}
        self._scoped_knowledge: dict[str, KnowledgeStore] = {}
        self._scoped_knowledge_sets: dict[tuple[str, ...], KnowledgeStore] = {}
        for card in cards:
            scoped = KnowledgeStore(root=self._knowledge_root)
            scoped.cards = [card]
            scoped.version = self.knowledge.version
            self._scoped_knowledge[card.id] = scoped
            self._scoped_knowledge_sets[(card.id,)] = scoped

        alias_entries: dict[str, set[str]] = {}
        for entry in self.entries.values():
            if entry.review_status != "source_checked":
                continue
            for term in {entry.name, *entry.aliases}:
                normalized = term.strip().casefold()
                if (
                    len(normalized) < 2
                    or normalized in _GENERIC_TOPIC_TERMS
                    or any(phrase in normalized for phrase in _GENERIC_TOPIC_PHRASES)
                ):
                    continue
                alias_entries.setdefault(normalized, set()).add(entry.id)
        self._unique_topic_terms = tuple(
            sorted(
                ((term, next(iter(entry_ids))) for term, entry_ids in alias_entries.items() if len(entry_ids) == 1),
                key=lambda item: (-len(item[0]), item[0], item[1]),
            )
        )

    def runtime_status(self) -> dict:
        """Report loaded content separately from disk; never replace active knowledge."""
        try:
            disk_sha256 = hashlib.sha256(self._catalog_path.read_bytes()).hexdigest()
        except OSError:
            disk_sha256 = None
        matches = disk_sha256 == self.loaded_sha256 if disk_sha256 is not None else None
        return {
            "loaded_version": self.knowledge.version,
            "loaded_sha256": self.loaded_sha256,
            "disk_sha256": disk_sha256,
            "disk_status": "unavailable" if matches is None else "current" if matches else "changed",
            "restart_required": None if matches is None else not matches,
            "source_count": len(self.sources),
            "published_image_count": sum(m.kind == "image" and m.publication_status == "published" for m in self.media.values()),
            "published_video_count": sum(m.kind == "video" and m.publication_status == "published" for m in self.media.values()),
        }

    def get(self, entry_id: str) -> EducationEntry | None:
        return self.entries.get(entry_id)

    def knowledge_for_entry(self, entry_id: str) -> KnowledgeStore | None:
        """Return a knowledge view containing only the selected education topic."""
        return self._scoped_knowledge.get(entry_id)

    def knowledge_for_entries(self, entry_ids: list[str] | tuple[str, ...]) -> KnowledgeStore | None:
        """Return a source-checked, version-bound view of bounded selected topics."""
        normalized_ids = tuple(dict.fromkeys(entry_ids))
        if not normalized_ids or len(normalized_ids) > EDUCATION_IMPORTED_CARD_LIMIT:
            return None
        if any(
            entry_id not in self.entries
            or self.entries[entry_id].review_status != "source_checked"
            or entry_id not in self._knowledge_cards
            for entry_id in normalized_ids
        ):
            return None
        if normalized_ids in self._scoped_knowledge_sets:
            return self._scoped_knowledge_sets[normalized_ids]
        scoped = KnowledgeStore(root=self._knowledge_root)
        scoped.cards = [self._knowledge_cards[entry_id] for entry_id in normalized_ids]
        scoped.version = self.knowledge.version
        self._scoped_knowledge_sets[normalized_ids] = scoped
        return scoped

    def related_entry_ids(self, text: str, *, excluding: str, limit: int = 2) -> list[str]:
        """Find only uniquely named, source-checked topics explicitly mentioned by the user."""
        query = text.casefold()
        scores: dict[str, int] = {}
        for term, entry_id in self._unique_topic_terms:
            if entry_id == excluding:
                continue
            if re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", term):
                matched = re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", query) is not None
            else:
                matched = term in query
            if matched:
                scores[entry_id] = max(scores.get(entry_id, 0), len(term))
        return sorted(scores, key=lambda entry_id: (-scores[entry_id], entry_id))[:max(0, limit)]

    def search(self, *, kind: str | None = None, query: str = "") -> list[EducationEntry]:
        needle = query.strip().casefold()
        values = [entry for entry in self.entries.values() if (kind is None or entry.kind == kind)]
        if needle:
            exact = [
                entry for entry in values
                if needle in {value.casefold() for value in (entry.name, *entry.aliases)}
            ]
            if exact:
                values = exact
            else:
                values = [entry for entry in values if needle in " ".join(
                    [entry.name, entry.category, entry.summary, *entry.aliases]
                ).casefold()]
        return sorted(values, key=lambda entry: ({"cancer": 0, "gene": 1, "concept": 2}[entry.kind], entry.category, entry.name))

    @staticmethod
    def scope_notice(*, kind: str | None = None, query: str = "") -> dict | None:
        normalized = re.sub(r"\s+", " ", query.strip().casefold())
        if not normalized:
            return None
        for notice in _EDUCATION_SCOPE_NOTICES:
            if kind is not None and kind != notice["kind"]:
                continue
            if not any(term in normalized for term in notice["terms"]):
                continue
            if any(term in normalized for term in notice["specific_terms"]):
                continue
            return {
                "title": notice["title"],
                "message": notice["message"],
                "sources": [dict(source) for source in notice["sources"]],
            }
        return None

    def public(self, entry: EducationEntry, *, detail: bool = False) -> dict:
        values = {
            "id": entry.id, "kind": entry.kind, "category": entry.category,
            "name": entry.name, "aliases": entry.aliases, "summary": entry.summary,
            "source_url": str(entry.source_url), "source_title": entry.source_title,
            "source_date": entry.source_date, "checked_at": entry.checked_at,
            "review_status": entry.review_status,
        }
        if detail:
            values["body"] = entry.body
            if entry.faqs:
                source_ids = list(dict.fromkeys(
                    source_id for faq in entry.faqs for source_id in faq.source_ids
                ))
                values["faqs"] = [faq.model_dump(mode="json") for faq in entry.faqs]
                values["faq_sources"] = [
                    self.sources[source_id].model_dump(mode="json") for source_id in source_ids
                ]
            values["media"] = [
                self.media[media_id].model_dump(mode="json")
                for media_id in entry.media_ids
                if self.media[media_id].publication_status == "published"
            ]
        return values


class EducationChat:
    """Topic-bound adapter over isolated in-memory chat sessions."""

    def __init__(self, catalog: EducationCatalog, chat_service: ResearchChatService):
        self.catalog = catalog
        self.chat_service = chat_service
        self._sessions: dict[UUID, tuple[str, str, tuple[str, ...]]] = {}
        self._imported_context: dict[UUID, tuple[str, ...]] = {}
        self._lock = RLock()

    def health(self) -> dict:
        return {
            "catalog": self.catalog.runtime_status(),
            "catalog_count": len(self.catalog.entries),
            "cancer_count": sum(e.kind == "cancer" for e in self.catalog.entries.values()),
            "gene_count": sum(e.kind == "gene" for e in self.catalog.entries.values()),
            "concept_count": sum(e.kind == "concept" for e in self.catalog.entries.values()),
            "chat": self.chat_service.health(),
        }

    def close(self):
        self.chat_service.close()

    def create_session(self, entry_id: str) -> dict:
        if self.catalog.get(entry_id) is None:
            raise ChatServiceError("education_entry_not_found", 404)
        result = self.chat_service.create_session()
        session_id = UUID(result["session_id"])
        with self._lock:
            knowledge_version = self.catalog.knowledge.version
            self._sessions[session_id] = (entry_id, knowledge_version, (entry_id,))
            self._imported_context[session_id] = ()
        return {**result, "entry_id": entry_id, "knowledge_version": knowledge_version}

    def submit(self, request: EducationMessageRequest):
        with self._lock:
            binding = self._sessions.get(request.session_id)
        if binding is None:
            raise ChatServiceError("session_not_found", 404)
        bound_entry, bound_version, previous_context_ids = binding
        if bound_entry != request.entry_id:
            raise ChatServiceError("education_topic_changed", 409)
        if bound_version != self.catalog.knowledge.version:
            raise ChatServiceError("education_knowledge_version_changed", 409)
        entry = self.catalog.get(bound_entry)
        if entry is None or entry.review_status != "source_checked":
            raise ChatServiceError("education_source_unavailable", 409)
        try:
            ensure_no_sensitive_query(request.text)
        except InvalidInputError:
            raise ChatServiceError("personal_medical_advice_or_sensitive_data", 422) from None
        related_ids = self.catalog.related_entry_ids(
            request.text, excluding=bound_entry,
            limit=EDUCATION_CHAT_CONTEXT_TOPIC_LIMIT - 1,
        )
        if request.context_entry_ids is None:
            explicit_ids = self._imported_context.get(request.session_id, ())
            removed_ids = set()
        else:
            explicit_ids = tuple(request.context_entry_ids)
            if len(set(explicit_ids)) != len(explicit_ids):
                raise ChatServiceError("education_context_invalid", 400)
            if len(set((bound_entry, *explicit_ids))) > EDUCATION_IMPORTED_CARD_LIMIT:
                raise ChatServiceError("education_context_limit", 400)
            if any(self.catalog.get(key) is None or self.catalog.get(key).review_status != "source_checked"
                   for key in explicit_ids):
                raise ChatServiceError("education_source_unavailable", 400)
            removed_ids = set(self._imported_context.get(request.session_id, ())) - set(explicit_ids)
        retained = [key for key in previous_context_ids if key not in removed_ids]
        limit = EDUCATION_IMPORTED_CARD_LIMIT if explicit_ids else EDUCATION_CHAT_CONTEXT_TOPIC_LIMIT
        context_ids = tuple(dict.fromkeys((bound_entry, *explicit_ids, *related_ids, *retained)))[:limit]
        scoped_knowledge = self.catalog.knowledge_for_entries(context_ids)
        if scoped_knowledge is None:
            raise ChatServiceError("education_source_unavailable", 409)
        matched_cards = scoped_knowledge.search(
            " ".join([entry.id, entry.name, *entry.aliases]), limit=4,
        )
        if not any(card.id == entry.id for card in matched_cards):
            raise ChatServiceError("education_source_unavailable", 409)
        if _personal_medical_request(request.text):
            raise ChatServiceError("personal_medical_advice_or_sensitive_data", 422)
        included_entries = [self.catalog.get(entry_id) for entry_id in context_ids]
        context_description = "；".join(f"{item.id}：{item.name}" for item in included_entries if item is not None)
        scoped_text = (
            f"当前科普主主题 ID：{entry.id}；名称：{entry.name}。"
            f"本轮允许使用的科普主题 ID 与名称：{context_description}。"
            f"只能依据这些已核对词条回答；用户明确提到的其他主题以外不扩展。"
            f"用户问题：{request.text.strip()}"
        )
        delegated = ChatMessageRequest(
            session_id=request.session_id,
            message_id=request.message_id,
            text=scoped_text,
            use_evidence=False,
            attachment_ids=request.attachment_ids,
            knowledge_ids=list(context_ids),
        )
        job = self.chat_service.submit(delegated, knowledge=scoped_knowledge, history_text=request.text)
        with self._lock:
            if request.session_id in self._sessions:
                self._sessions[request.session_id] = (bound_entry, bound_version, context_ids)
                self._imported_context[request.session_id] = explicit_ids
        return job

    def get_job(self, session_id: UUID, job_id: UUID):
        with self._lock:
            if session_id not in self._sessions:
                raise ChatServiceError("session_not_found", 404)
        return self.chat_service.get_job(session_id, job_id)

    def clear(self, session_id: UUID):
        with self._lock:
            if session_id not in self._sessions:
                raise ChatServiceError("session_not_found", 404)
            self.chat_service.clear(session_id)
            self._sessions.pop(session_id, None)
            self._imported_context.pop(session_id, None)
        return {"status": "cleared"}


_PERSONAL_CANCER_PATTERN = re.compile(
    r"(?:我|本人|家人|亲属)(?:被诊断|确诊|患有|得了).{0,30}(?:癌|肿瘤)|"
    r"(?:我|本人|家人|亲属).{0,20}(?:怎么办|怎么治疗|怎么治|吃什么药|用什么药|治疗方案)",
    re.IGNORECASE,
)


def _personal_medical_request(text: str) -> bool:
    return bool(_PERSONAL_CANCER_PATTERN.search(text))


EDUCATION_SYSTEM_PROMPT = """你是基因与癌症的中文科普助手，只解释当前用户选定的词条及与其直接相关的基础概念。
只把当前提供的科普知识卡作为已核实事实；回答必须引用允许的知识卡 ID，使用 [K-edu-...] 标记，JSON 中 citation_ids 只列出同一批 ID。
不把基因、蛋白和变异混为一谈；不把关联写成因果，不将群体研究结论套用到个人，不提供个体诊断、风险评估、治疗或用药建议。
遇到个人病情/用药问题时说明本助手不能据此判断，并建议向合格医疗专业人员咨询。偏离基因或癌症科普的问题简短说明范围并引导回当前词条。
资料没有覆盖的问题要明确说当前词条资料不足，不要编造结论、出处或“最新研究”。可提出后续学习方向，但须标为建议而非已核实事实。
保持简明中文，默认先用 2–4 句回答核心问题；必要时补一个有资料支持的例子，不强行编比喻。用户明确要求详细解释时才展开，不自行增加诊疗建议。保留 TP53、BRCA1 等标准基因符号。不要输出 URL，页面会提供来源链接。只返回单个 JSON 对象，字段严格为 answer（中文字符串）和 citation_ids（本次提供 ID 的数组）。
有资料支持的事实至少引用一条资料。正文每个引用都使用英文半角方括号包住本次 allowed_source_ids 中的完整 ID；不要只写裸 ID、圆括号或中文括号。
citation_ids 必须与正文方括号内引用 ID 的集合完全相同；正文没写出的 ID 不放入数组，正文写出的每个 ID 都放入数组。
同一个 ID 可在正文多次引用，数组中只列一次；不要把占位符 K-edu-... 写入回答。回答最多 600 个汉字。"""

