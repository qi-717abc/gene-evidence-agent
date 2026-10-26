"""Deterministic concurrent collection from PubMed, UniProt, and Open Targets."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from gene_evidence_agent.aggregation import aggregate_evidence
from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.disease_terms import (
    UnsupportedChineseDiseaseTerm,
    contains_chinese_text,
    normalize_disease_search_term,
)
from gene_evidence_agent.errors import InvalidInputError, ToolError
from gene_evidence_agent.schemas import (
    BiomedicalEntity,
    EvidenceRecord,
    EvidenceReport,
    SourceOutcome,
    SourceReference,
    StrictModel,
)
from gene_evidence_agent.tools.opentargets import OpenTargetsAssociationResult, OpenTargetsClient, OpenTargetsSearchResult
from gene_evidence_agent.tools.pubmed import PubMedClient, PubMedSearchResult, parse_pubmed_evidence
from gene_evidence_agent.tools.pubmed_query import build_gene_disease_query
from gene_evidence_agent.tools.uniprot import UniProtClient, UniProtSearchResult

_MAX_RESULTS = 50
_SOURCE_NAMES = ("pubmed", "uniprot", "opentargets")


class EvidenceCollection(StrictModel):
    """Raw source results gathered for one gene-disease request."""

    gene: BiomedicalEntity
    disease: BiomedicalEntity
    pubmed_search: PubMedSearchResult | None = None
    pubmed_evidence: list[EvidenceRecord] = Field(default_factory=list, max_length=_MAX_RESULTS)
    evidence_records: list[EvidenceRecord] = Field(default_factory=list, max_length=200)
    uniprot: UniProtSearchResult | None = None
    opentargets_target_search: OpenTargetsSearchResult | None = None
    opentargets_disease_search: OpenTargetsSearchResult | None = None
    opentargets_association: OpenTargetsAssociationResult | None = None
    source_outcomes: list[SourceOutcome] = Field(min_length=3, max_length=3)
    warnings: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    request_id: UUID

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    def to_evidence_report(self, *, query: str | None = None) -> EvidenceReport:
        """Build a source-grounded report without invoking a language model."""

        reported_gene = _resolve_query_entity(self.gene, self.opentargets_target_search)
        reported_disease = _resolve_query_entity(self.disease, self.opentargets_disease_search)
        entities_by_id = {reported_gene.id: reported_gene, reported_disease.id: reported_disease}
        for search_result in (self.opentargets_target_search, self.opentargets_disease_search):
            if search_result is not None:
                for entity in search_result.to_biomedical_entities():
                    entities_by_id.setdefault(entity.id, entity)
        if self.opentargets_association is not None:
            for candidate in self.opentargets_association.clinical_drug_candidates:
                chembl_id = candidate.drug_id if re.fullmatch(r"CHEMBL[0-9]+", candidate.drug_id) else None
                entities_by_id.setdefault(
                    f"opentargets:drug:{candidate.drug_id}",
                    BiomedicalEntity(
                        id=f"opentargets:drug:{candidate.drug_id}",
                        entity_type="drug",
                        query_name=candidate.drug_name,
                        canonical_name=candidate.drug_name,
                        identifier=chembl_id,
                        namespace="ChEMBL" if chembl_id else None,
                        resolution_status="resolved" if chembl_id else "unresolved",
                    ),
                )

        evidence_gaps: list[str] = []
        for outcome in self.source_outcomes:
            if outcome.status not in {"success", "not_requested"}:
                evidence_gaps.append(f"{outcome.source} source status: {outcome.status}")
        limitations = [
            "Database associations and annotations are evidence records, not by themselves proof of causation.",
            "Clinical-stage information does not establish efficacy or suitability for an individual.",
            "This deterministic report was generated without an external language model summary.",
        ]
        return EvidenceReport(
            query=query or f"Gene {self.gene.query_name}; disease {self.disease.query_name}",
            entities=list(entities_by_id.values()),
            evidence=self.evidence_records,
            source_outcomes=self.source_outcomes,
            warnings=self.warnings,
            evidence_gaps=evidence_gaps,
            limitations=limitations,
            generated_at=self.retrieved_at,
        )

    def to_json_report(self, *, query: str | None = None) -> str:
        """Return the deterministic evidence report as formatted JSON text."""

        return self.to_evidence_report(query=query).model_dump_json(indent=2, exclude_none=True)


def _resolve_query_entity(
    query_entity: BiomedicalEntity,
    search_result: OpenTargetsSearchResult | None,
) -> BiomedicalEntity:
    """Attach a uniquely selected Open Targets identifier to the original query ID."""

    if search_result is None or not search_result.hits:
        return query_entity
    selected_hit = (
        search_result.hits[0]
        if search_result.status == "success"
        else search_result.unique_exact_match()
    )
    if selected_hit is None:
        return query_entity
    hit_index = search_result.hits.index(selected_hit)
    resolved = search_result.to_biomedical_entities()[hit_index]
    if resolved.resolution_status != "resolved" or resolved.identifier is None or resolved.namespace is None:
        return query_entity
    return BiomedicalEntity(
        id=query_entity.id,
        entity_type=query_entity.entity_type,
        query_name=query_entity.query_name,
        canonical_name=resolved.canonical_name,
        identifier=resolved.identifier,
        namespace=resolved.namespace,
        organism_tax_id=query_entity.organism_tax_id,
        resolution_status="resolved",
    )


class _OpenTargetsCollection(StrictModel):
    target_search: OpenTargetsSearchResult | None = None
    disease_search: OpenTargetsSearchResult | None = None
    association: OpenTargetsAssociationResult | None = None
    warnings: list[str] = Field(default_factory=list)
    outcome: SourceOutcome


class GeneEvidencePipeline:
    """Run each source branch concurrently and preserve partial results."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pubmed: Any | None = None,
        uniprot: Any | None = None,
        opentargets: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pubmed = pubmed or PubMedClient(self._settings)
        self._uniprot = uniprot or UniProtClient(self._settings)
        self._opentargets = opentargets or OpenTargetsClient(self._settings)

    async def collect(
        self,
        gene: BiomedicalEntity,
        disease: BiomedicalEntity,
        *,
        limit: int | None = None,
        sources: set[str] | None = None,
        include_drugs: bool = True,
    ) -> EvidenceCollection:
        """Fetch selected source groups concurrently without merging records yet.

        Omitting ``sources`` preserves the historical all-source behavior.
        Unselected sources are represented explicitly as ``not_requested``.
        """

        if gene.entity_type != "gene" or disease.entity_type != "disease":
            raise InvalidInputError("Pipeline requires one gene and one disease entity", source="pipeline")
        if gene.id == disease.id:
            raise InvalidInputError("Gene and disease must have distinct internal entity IDs", source="pipeline")
        if contains_chinese_text(gene.query_name):
            raise InvalidInputError(
                "Gene search terms must use an English gene symbol",
                source="pipeline",
            )
        try:
            disease_term = normalize_disease_search_term(disease.query_name)
        except UnsupportedChineseDiseaseTerm as error:
            raise InvalidInputError(
                "Chinese disease terms must have a known English mapping; provide the English disease name",
                source="pipeline",
            ) from error
        source_disease = disease.model_copy(update={"query_name": disease_term.search_term})
        reported_disease = disease.model_copy(
            update={
                "canonical_name": disease_term.search_term
                if disease_term.translated
                else disease.canonical_name
            }
        )
        result_limit = self._settings.max_results_per_source if limit is None else limit
        if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= _MAX_RESULTS:
            raise InvalidInputError(f"Pipeline result limit must be between 1 and {_MAX_RESULTS}", source="pipeline")
        selected_sources = set(_SOURCE_NAMES) if sources is None else set(sources)
        if not selected_sources or not selected_sources.issubset(_SOURCE_NAMES):
            raise InvalidInputError("Pipeline sources must select one or more supported sources", source="pipeline")
        if not isinstance(include_drugs, bool):
            raise InvalidInputError("include_drugs must be a boolean", source="pipeline")

        tasks: dict[str, Any] = {}
        if "pubmed" in selected_sources:
            tasks["pubmed"] = self._collect_pubmed(gene, source_disease, result_limit)
        if "uniprot" in selected_sources:
            tasks["uniprot"] = self._uniprot.search_human_gene(gene, limit=result_limit)
        if "opentargets" in selected_sources:
            tasks["opentargets"] = self._collect_opentargets(
                gene, source_disease, result_limit, include_drugs=include_drugs
            )
        attempts = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results = dict(zip(tasks.keys(), attempts, strict=True))

        warnings: list[str] = ["disease_query_translated_from_chinese"] if disease_term.translated else []
        outcomes: list[SourceOutcome] = []
        source_records: list[EvidenceRecord] = []

        if "pubmed" not in selected_sources:
            pubmed_search, pubmed_evidence = None, []
            outcomes.append(SourceOutcome(source="pubmed", status="not_requested"))
        else:
            pubmed_attempt = results["pubmed"]
            if isinstance(pubmed_attempt, BaseException):
                _raise_if_cancelled(pubmed_attempt)
                pubmed_search, pubmed_evidence = None, []
                outcomes.append(_failed_outcome("pubmed", pubmed_attempt))
                warnings.append(f"pubmed_{_error_code(pubmed_attempt)}")
            else:
                pubmed_search, pubmed_evidence = pubmed_attempt
                outcomes.append(
                    SourceOutcome(
                        source="pubmed",
                        status="success" if pubmed_evidence else "empty",
                        record_count=len(pubmed_evidence),
                    )
                )
                source_records.extend(pubmed_evidence)

        if "uniprot" not in selected_sources:
            uniprot_result = None
            outcomes.append(SourceOutcome(source="uniprot", status="not_requested"))
        else:
            uniprot_attempt = results["uniprot"]
            if isinstance(uniprot_attempt, BaseException):
                _raise_if_cancelled(uniprot_attempt)
                uniprot_result = None
                outcomes.append(_failed_outcome("uniprot", uniprot_attempt))
                warnings.append(f"uniprot_{_error_code(uniprot_attempt)}")
            else:
                uniprot_result = uniprot_attempt
                outcomes.append(
                    SourceOutcome(
                        source="uniprot",
                        status=uniprot_result.status,
                        record_count=len(uniprot_result.entries),
                    )
                )
                source_records.extend(_uniprot_evidence(gene, uniprot_result))

        if "opentargets" not in selected_sources:
            opentargets_result = _OpenTargetsCollection(
                warnings=[], outcome=SourceOutcome(source="opentargets", status="not_requested")
            )
        else:
            opentargets_attempt = results["opentargets"]
            if isinstance(opentargets_attempt, BaseException):
                _raise_if_cancelled(opentargets_attempt)
                opentargets_result = _OpenTargetsCollection(
                    warnings=[], outcome=_failed_outcome("opentargets", opentargets_attempt)
                )
                warnings.append(f"opentargets_{_error_code(opentargets_attempt)}")
            else:
                opentargets_result = opentargets_attempt
                warnings.extend(opentargets_result.warnings)
        outcomes.append(opentargets_result.outcome)

        if opentargets_result.association is not None:
            source_records.extend(opentargets_result.association.to_evidence_records())
        return EvidenceCollection(
            gene=gene,
            disease=reported_disease,
            pubmed_search=pubmed_search,
            pubmed_evidence=pubmed_evidence,
            evidence_records=aggregate_evidence(source_records),
            uniprot=uniprot_result,
            opentargets_target_search=opentargets_result.target_search,
            opentargets_disease_search=opentargets_result.disease_search,
            opentargets_association=opentargets_result.association,
            source_outcomes=outcomes,
            warnings=list(dict.fromkeys(warnings)),
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )

    async def _collect_pubmed(
        self, gene: BiomedicalEntity, disease: BiomedicalEntity, limit: int
    ) -> tuple[PubMedSearchResult, list[EvidenceRecord]]:
        query = build_gene_disease_query(gene, disease)
        search_result = await self._pubmed.search(query, limit=limit)
        if not search_result.pmids:
            return search_result, []
        fetched = await self._pubmed.fetch_xml(search_result.pmids)
        evidence = parse_pubmed_evidence(fetched, entity_ids=[gene.id, disease.id])
        return search_result, evidence

    async def _collect_opentargets(
        self, gene: BiomedicalEntity, disease: BiomedicalEntity, limit: int, *, include_drugs: bool
    ) -> _OpenTargetsCollection:
        target_attempt, disease_attempt = await asyncio.gather(
            self._opentargets.search_entities(gene.query_name, entity_type="target", limit=limit),
            self._opentargets.search_entities(disease.query_name, entity_type="disease", limit=limit),
            return_exceptions=True,
        )
        warnings: list[str] = []
        target_search = None if isinstance(target_attempt, BaseException) else target_attempt
        disease_search = None if isinstance(disease_attempt, BaseException) else disease_attempt
        errors = [result for result in (target_attempt, disease_attempt) if isinstance(result, BaseException)]
        for error in errors:
            _raise_if_cancelled(error)
            warnings.append(f"opentargets_entity_search_{_error_code(error)}")
        if errors and (target_search is None or disease_search is None):
            count = sum(len(result.hits) for result in (target_search, disease_search) if result is not None)
            outcome = SourceOutcome(
                source="opentargets",
                status="partial_success" if count else _error_code(errors[0]),
                record_count=count,
                error_code=_error_code(errors[0]),
            )
            return _OpenTargetsCollection(
                target_search=target_search,
                disease_search=disease_search,
                warnings=warnings,
                outcome=outcome,
            )
        assert target_search is not None and disease_search is not None
        target_hit = target_search.hits[0] if target_search.status == "success" else target_search.unique_exact_match()
        disease_hit = disease_search.hits[0] if disease_search.status == "success" else disease_search.unique_exact_match()
        if target_hit is None or disease_hit is None:
            warnings.append("opentargets_association_skipped_due_to_empty_or_ambiguous_entity_search")
            status = "ambiguous" if "ambiguous" in {target_search.status, disease_search.status} else "empty"
            count = len(target_search.hits) + len(disease_search.hits)
            return _OpenTargetsCollection(
                target_search=target_search,
                disease_search=disease_search,
                warnings=warnings,
                outcome=SourceOutcome(source="opentargets", status=status, record_count=count),
            )

        if target_search.status == "ambiguous":
            warnings.append("opentargets_target_resolved_by_unique_exact_name_match")
        if disease_search.status == "ambiguous":
            warnings.append("opentargets_disease_resolved_by_unique_exact_name_match")
        target_entity = target_search.to_biomedical_entities()[target_search.hits.index(target_hit)]
        disease_entity = disease_search.to_biomedical_entities()[disease_search.hits.index(disease_hit)]
        if target_entity.resolution_status != "resolved" or target_entity.namespace != "Ensembl":
            warnings.append("opentargets_association_skipped_due_to_unresolved_target_identifier")
            return _OpenTargetsCollection(
                target_search=target_search,
                disease_search=disease_search,
                warnings=warnings,
                outcome=SourceOutcome(source="opentargets", status="partial_success", record_count=2),
            )
        if disease_entity.resolution_status != "resolved" or disease_entity.namespace not in {"EFO", "MONDO"}:
            warnings.append("opentargets_association_skipped_due_to_noncanonical_disease_identifier")
            return _OpenTargetsCollection(
                target_search=target_search,
                disease_search=disease_search,
                warnings=warnings,
                outcome=SourceOutcome(source="opentargets", status="partial_success", record_count=2),
            )
        try:
            association = await self._opentargets.get_target_disease_association(
                target_entity.identifier,
                disease_entity.identifier,
                limit=limit,
                include_drugs=include_drugs,
            )
        except Exception as error:
            return _OpenTargetsCollection(
                target_search=target_search,
                disease_search=disease_search,
                warnings=[f"opentargets_association_{_error_code(error)}"],
                outcome=SourceOutcome(
                    source="opentargets",
                    status="partial_success",
                    record_count=len(target_search.hits) + len(disease_search.hits),
                    error_code=_error_code(error),
                ),
            )
        status = "empty" if association.status in {"empty", "target_not_found"} else "success"
        if not include_drugs and association.clinical_drug_candidates:
            association = association.model_copy(update={"clinical_drug_candidates": [], "clinical_drug_total_count": 0})
        warnings.extend(association.warnings)
        count = len(target_search.hits) + len(disease_search.hits) + len(association.associations) + len(association.clinical_drug_candidates)
        return _OpenTargetsCollection(
            target_search=target_search,
            disease_search=disease_search,
            association=association,
            warnings=warnings,
            outcome=SourceOutcome(source="opentargets", status=status, record_count=count),
        )


def _uniprot_evidence(gene: BiomedicalEntity, result: UniProtSearchResult) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for candidate in result.entries:
        sections = [
            f"UniProtKB maps {gene.query_name} to protein accession {candidate.accession} "
            f"({'reviewed' if candidate.reviewed else 'unreviewed'} record)."
        ]
        warnings: list[str] = []
        if candidate.function_annotations:
            sections.append(f"Function annotation: {candidate.function_annotations[0]}")
            if len(candidate.function_annotations) > 1:
                warnings.append("additional_function_annotations_available_in_source_result")
        if candidate.disease_annotations:
            sections.append(f"Disease annotation: {candidate.disease_annotations[0]}")
            if len(candidate.disease_annotations) > 1:
                warnings.append("additional_disease_annotations_available_in_source_result")
        records.append(
            EvidenceRecord(
                evidence_type="protein_annotation",
                source=SourceReference(
                    source_database="UniProtKB",
                    record_id=candidate.accession,
                    record_url=candidate.record_url,
                    retrieved_at=result.retrieved_at,
                ),
                entity_ids=[gene.id],
                title=candidate.protein_name or candidate.entry_name or candidate.accession,
                summary=" ".join(sections),
                warnings=warnings,
            )
        )
    return records


def _error_code(error: BaseException) -> Literal["invalid_input", "rate_limited", "timeout", "upstream_error"]:
    if isinstance(error, ToolError):
        value = error.status.value
        if value in {"invalid_input", "rate_limited", "timeout", "upstream_error"}:
            return value  # type: ignore[return-value]
    return "upstream_error"


def _failed_outcome(source: Literal["pubmed", "uniprot", "opentargets"], error: BaseException) -> SourceOutcome:
    code = _error_code(error)
    return SourceOutcome(source=source, status=code, error_code=code)


def _raise_if_cancelled(error: BaseException) -> None:
    if isinstance(error, asyncio.CancelledError):
        raise error
