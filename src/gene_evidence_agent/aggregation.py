"""Stable evidence deduplication and provenance aggregation."""

from __future__ import annotations

import re

from gene_evidence_agent.schemas import EvidenceRecord, SourceReference

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def aggregate_evidence(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Deduplicate by PMID/DOI or source record, retain first-seen order and all provenance."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    alias_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        aliases = _identity_aliases(record)
        for alias in aliases:
            previous = alias_owner.get(alias)
            if previous is None:
                alias_owner[alias] = index
            else:
                union(index, previous)

    grouped: dict[int, list[EvidenceRecord]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)

    aggregated: list[EvidenceRecord] = []
    for group in grouped.values():
        primary = group[0]
        also_reported_by: list[SourceReference] = list(primary.also_reported_by)
        seen_sources = {_source_fingerprint(primary.source)}
        seen_sources.update(_source_fingerprint(source) for source in also_reported_by)
        entity_ids = list(primary.entity_ids)
        warnings = list(primary.warnings)
        for duplicate in group[1:]:
            fingerprint = _source_fingerprint(duplicate.source)
            if fingerprint not in seen_sources:
                also_reported_by.append(duplicate.source)
                seen_sources.add(fingerprint)
            for source in duplicate.also_reported_by:
                fingerprint = _source_fingerprint(source)
                if fingerprint not in seen_sources:
                    also_reported_by.append(source)
                    seen_sources.add(fingerprint)
            entity_ids.extend(entity_id for entity_id in duplicate.entity_ids if entity_id not in entity_ids)
            warnings.extend(warning for warning in duplicate.warnings if warning not in warnings)
            if duplicate.summary != primary.summary and "duplicate_content_conflict" not in warnings:
                warnings.append("duplicate_content_conflict")
            if duplicate.score != primary.score and "duplicate_score_conflict_primary_score_retained" not in warnings:
                warnings.append("duplicate_score_conflict_primary_score_retained")
        if len(also_reported_by) > 20:
            warnings.append("provenance_sources_truncated_to_20")
        aggregated.append(
            primary.model_copy(
                update={
                    "entity_ids": entity_ids,
                    "also_reported_by": also_reported_by[:20],
                    "warnings": warnings,
                }
            )
        )
    return aggregated


def _identity_aliases(record: EvidenceRecord) -> set[str]:
    aliases: set[str] = set()
    if record.pmid:
        aliases.add(f"{record.evidence_type}:pmid:{record.pmid}")
    if record.doi:
        doi = _DOI_PREFIX.sub("", record.doi.strip()).rstrip(".,; )").casefold()
        if doi:
            aliases.add(f"{record.evidence_type}:doi:{doi}")
    if not aliases:
        aliases.add(
            f"{record.evidence_type}:record:{record.source.source_database.casefold()}:{record.source.record_id.casefold()}"
        )
    return aliases


def _source_fingerprint(source: SourceReference) -> tuple[str, str, str]:
    return (source.source_database.casefold(), source.record_id.casefold(), str(source.record_url))
