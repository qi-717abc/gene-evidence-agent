"""Build conservative, deterministic PubMed gene-disease searches."""

from __future__ import annotations

import unicodedata

from gene_evidence_agent.errors import InvalidInputError
from gene_evidence_agent.schemas import BiomedicalEntity

_QUERY_SYNTAX_CHARACTERS = frozenset('"[]()&|:#*')
_MAX_TERM_LENGTH = 200
_SOURCE = "pubmed"


def _format_entity_terms(entity: BiomedicalEntity, expected_type: str) -> str:
    if entity.entity_type != expected_type:
        raise InvalidInputError(
            f"Expected a {expected_type} entity, got {entity.entity_type}",
            source=_SOURCE,
        )

    aliases: list[str] = []
    for raw_term in (entity.query_name, entity.canonical_name):
        if raw_term is None:
            continue
        term = " ".join(unicodedata.normalize("NFKC", raw_term).split())
        if not term:
            continue
        if len(term) > _MAX_TERM_LENGTH or any(
            character in _QUERY_SYNTAX_CHARACTERS for character in term
        ):
            raise InvalidInputError(
                f"The {expected_type} search term contains unsupported characters or is too long",
                source=_SOURCE,
            )
        if term.casefold() not in {alias.casefold() for alias in aliases}:
            aliases.append(term)

    if not aliases:
        raise InvalidInputError(
            f"The {expected_type} entity has no searchable name",
            source=_SOURCE,
        )

    return " OR ".join(f'"{term}"[Title/Abstract]' for term in aliases)


def build_gene_disease_query(gene: BiomedicalEntity, disease: BiomedicalEntity) -> str:
    """Build a PubMed query from the supplied entity names, without free syntax.

    Query and canonical names are treated as quoted phrases in the Title/Abstract
    field. This prevents entity text from becoming executable PubMed query
    operators while retaining a simple alias-expansion strategy.
    """

    gene_terms = _format_entity_terms(gene, "gene")
    disease_terms = _format_entity_terms(disease, "disease")
    return f"({gene_terms}) AND ({disease_terms})"
