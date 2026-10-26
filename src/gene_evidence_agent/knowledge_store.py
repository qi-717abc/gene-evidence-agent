"""Small source-checked knowledge collection with deterministic alias retrieval."""

import hashlib
import json
import re
from pathlib import Path

from pydantic import Field, field_validator

from gene_evidence_agent.chat_schemas import ChatCitation
from gene_evidence_agent.schemas import StrictModel

KNOWLEDGE_ROOT = Path(__file__).with_name("knowledge") / "gene_cancer"


class KnowledgeExcerpt(StrictModel):
    citation: ChatCitation
    text: str = Field(min_length=1, max_length=8000)


class KnowledgeCard(StrictModel):
    id: str = Field(pattern=r"^K-[a-z0-9-]{1,70}$")
    title: str = Field(min_length=1, max_length=300)
    keywords: list[str] = Field(min_length=1, max_length=30)
    source_url: str
    source_title: str
    source_date: str | None
    checked_at: str
    review_status: str = Field(pattern=r"^(draft|source_checked)$")
    scope: str
    body_file: str = Field(pattern=r"^[a-z0-9-]+\.md$")
    body: str = ""
    excerpts: list[KnowledgeExcerpt] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, values):
        if any(not v.strip() for v in values):
            raise ValueError("keywords must not be blank")
        return values


class KnowledgeStore:
    def __init__(self, root: Path = KNOWLEDGE_ROOT):
        index = root / "index.json"
        self.cards: list[KnowledgeCard] = []
        self.version = "empty-v1"
        if not index.exists():
            return
        raw = index.read_bytes()
        data = json.loads(raw)
        digest = hashlib.sha256(raw)
        seen = set()
        for row in data["cards"]:
            card = KnowledgeCard.model_validate(row)
            if card.id in seen:
                raise ValueError("duplicate knowledge card id")
            seen.add(card.id)
            body = (root / card.body_file).read_text(encoding="utf-8").strip()
            if not body or len(body) > 4000:
                raise ValueError("invalid knowledge body")
            digest.update(body.encode("utf-8"))
            ChatCitation(source_id=card.id, kind="knowledge", title=card.title, url=card.source_url)
            if card.review_status == "source_checked":
                self.cards.append(card.model_copy(update={"body": body}))
        self.version = digest.hexdigest()[:16]

    def search(self, text: str, limit: int = 4) -> list[KnowledgeCard]:
        text = text.casefold()
        scored = []
        for card in self.cards:
            score = 0
            for word in card.keywords:
                needle = word.casefold()
                hit = (re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text)
                       if re.fullmatch(r"[a-z0-9 _-]+", needle) else needle in text)
                if hit:
                    score += max(1, len(needle))
            if score:
                scored.append((score, card.id, card))
        return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]
