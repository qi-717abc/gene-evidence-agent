"""Bounded target and disease entity search for the Open Targets Platform API."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import Field, field_validator, model_validator

from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import InvalidInputError, RateLimitedError, ToolTimeoutError, UpstreamError
from gene_evidence_agent.http_client import create_async_client
from gene_evidence_agent.schemas import BiomedicalEntity, EvidenceRecord, SourceReference, StrictModel

_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_MAX_RESULTS = 50
_SUPPORTED_IDENTIFIERS = {
    "target": (re.compile(r"ENSG[0-9]{11}"), "Ensembl", "gene"),
    "disease": (re.compile(r"EFO_[0-9]{7}"), "EFO", "disease"),
}
_SEARCH_QUERY = """
query SearchEntities($queryString: String!, $entityNames: [String!], $size: Int!) {
  search(
    queryString: $queryString
    entityNames: $entityNames
    page: {index: 0, size: $size}
  ) {
    total
    hits { id entity name description score }
  }
}
"""
_ASSOCIATION_QUERY = """
query TargetDiseaseAssociation($targetId: String!, $diseaseId: String!, $size: Int!) {
  target(ensemblId: $targetId) {
    id
    approvedSymbol
    associatedDiseases(
      Bs: [$diseaseId]
      enableIndirect: false
      page: {index: 0, size: $size}
    ) {
      count
      rows {
        score
        disease { id name }
        datasourceScores { id score }
        datatypeScores { id score }
      }
    }
  }
  disease(efoId: $diseaseId) {
    id
    drugAndClinicalCandidates {
      count
      rows { id maxClinicalStage drug { id name } }
    }
  }
}
"""
_ASSOCIATION_QUERY_NO_DRUGS = """
query TargetDiseaseAssociation($targetId: String!, $diseaseId: String!, $size: Int!) {
  target(ensemblId: $targetId) {
    id
    approvedSymbol
    associatedDiseases(
      Bs: [$diseaseId]
      enableIndirect: false
      page: {index: 0, size: $size}
    ) {
      count
      rows {
        score
        disease { id name }
        datasourceScores { id score }
        datatypeScores { id score }
      }
    }
  }
}
"""


def _normalize_label(value: str) -> str:
    """Normalize only casing and whitespace for conservative exact matching."""

    return " ".join(value.casefold().split())


class OpenTargetsSearchHit(StrictModel):
    """One target or disease returned by Open Targets full-text search."""

    entity_type: Literal["target", "disease"]
    identifier: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=2000)
    score: float | None = None


class OpenTargetsSearchResult(StrictModel):
    """Validated bounded search result envelope with source provenance."""

    query: str = Field(min_length=1, max_length=300)
    entity_type: Literal["target", "disease"]
    hits: list[OpenTargetsSearchHit] = Field(default_factory=list, max_length=_MAX_RESULTS)
    total_count: int = Field(ge=0)
    status: Literal["success", "empty", "ambiguous"]
    warnings: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    request_id: UUID

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> OpenTargetsSearchResult:
        expected_status = "empty" if not self.hits else ("success" if len(self.hits) == 1 else "ambiguous")
        if self.status != expected_status:
            raise ValueError(f"status must be {expected_status!r} for this hit list")
        if self.total_count < len(self.hits):
            raise ValueError("total_count cannot be less than the returned hit count")
        return self

    def to_biomedical_entities(self) -> list[BiomedicalEntity]:
        """Convert every hit to the common model without discarding unresolved IDs."""

        entities: list[BiomedicalEntity] = []
        for hit in self.hits:
            pattern, namespace, biomedical_type = _SUPPORTED_IDENTIFIERS[hit.entity_type]
            resolved = bool(pattern.fullmatch(hit.identifier))
            if hit.entity_type == "disease" and hit.identifier.startswith("MONDO_"):
                resolved = bool(re.fullmatch(r"MONDO_[0-9]{7}", hit.identifier))
                namespace = "MONDO"
            entities.append(
                BiomedicalEntity(
                    id=f"opentargets:{hit.entity_type}:{hit.identifier}",
                    entity_type=biomedical_type,
                    query_name=hit.name,
                    canonical_name=hit.name,
                    identifier=hit.identifier if resolved else None,
                    namespace=namespace if resolved else None,
                    resolution_status="resolved" if resolved else "unresolved",
                )
            )
        return entities

    def unique_exact_match(self) -> OpenTargetsSearchHit | None:
        """Return a sole case-insensitive exact-label hit, never a score-based guess."""

        normalized_query = _normalize_label(self.query)
        matches = [hit for hit in self.hits if _normalize_label(hit.name) == normalized_query]
        return matches[0] if len(matches) == 1 else None


class OpenTargetsScoreComponent(StrictModel):
    """One source or evidence-type contribution returned by Open Targets."""

    identifier: str = Field(min_length=1, max_length=200)
    score: float = Field(ge=0, le=1)


class OpenTargetsDiseaseAssociation(StrictModel):
    """Raw target-disease association fields prior to evidence conversion."""

    score: float = Field(ge=0, le=1)
    disease_id: str = Field(min_length=1, max_length=160)
    disease_name: str = Field(min_length=1, max_length=500)
    datasource_scores: list[OpenTargetsScoreComponent] = Field(default_factory=list, max_length=100)
    datatype_scores: list[OpenTargetsScoreComponent] = Field(default_factory=list, max_length=100)


class OpenTargetsClinicalDrugCandidate(StrictModel):
    """Drug-disease clinical-indication summary published by Open Targets."""

    indication_id: str = Field(min_length=1, max_length=200)
    disease_id: str = Field(min_length=1, max_length=160)
    drug_id: str = Field(min_length=1, max_length=160)
    drug_name: str = Field(min_length=1, max_length=500)
    max_clinical_stage: str = Field(min_length=1, max_length=100)


class OpenTargetsAssociationResult(StrictModel):
    """Bounded association response with target identity and source components."""

    target_id: str = Field(min_length=1, max_length=160)
    target_symbol: str | None = Field(default=None, max_length=100)
    disease_id: str = Field(min_length=1, max_length=160)
    associations: list[OpenTargetsDiseaseAssociation] = Field(default_factory=list, max_length=_MAX_RESULTS)
    clinical_drug_candidates: list[OpenTargetsClinicalDrugCandidate] = Field(default_factory=list, max_length=_MAX_RESULTS)
    clinical_drug_total_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    status: Literal["success", "empty", "target_not_found"]
    retrieved_at: datetime
    request_id: UUID

    @model_validator(mode="after")
    def validate_status(self) -> OpenTargetsAssociationResult:
        expected = "success" if self.associations else ("empty" if self.target_symbol else "target_not_found")
        if self.status != expected:
            raise ValueError(f"status must be {expected!r} for this target and association list")
        if self.total_count < len(self.associations):
            raise ValueError("total_count cannot be less than the returned association count")
        return self

    def to_evidence_records(self) -> list[EvidenceRecord]:
        """Create source-linked records without converting association into causality."""

        retrieved_at = self.retrieved_at
        records: list[EvidenceRecord] = []
        for association in self.associations:
            datasource_summary = ", ".join(
                f"{component.identifier}={component.score:.3f}" for component in association.datasource_scores
            ) or "not reported"
            datatype_summary = ", ".join(
                f"{component.identifier}={component.score:.3f}" for component in association.datatype_scores
            ) or "not reported"
            records.append(
                EvidenceRecord(
                    evidence_type="target_disease_association",
                    source=SourceReference(
                        source_database="Open Targets Platform",
                        record_id=f"{self.target_id}:{association.disease_id}",
                        record_url=f"https://platform.opentargets.org/target/{self.target_id}",
                        retrieved_at=retrieved_at,
                    ),
                    entity_ids=[f"opentargets:target:{self.target_id}", f"opentargets:disease:{association.disease_id}"],
                    title=f"{self.target_symbol or self.target_id} — {association.disease_name}",
                    summary=(
                        f"Open Targets reports an association score of {association.score:.3f}. "
                        f"Datasource scores: {datasource_summary}. Datatype scores: {datatype_summary}. "
                        "This database association score is not, by itself, proof of causation."
                    ),
                    score=association.score,
                )
            )
        for candidate in self.clinical_drug_candidates:
            records.append(
                EvidenceRecord(
                    evidence_type="disease_drug_candidate",
                    source=SourceReference(
                        source_database="Open Targets Platform",
                        record_id=candidate.indication_id,
                        record_url=f"https://platform.opentargets.org/disease/{candidate.disease_id}",
                        retrieved_at=retrieved_at,
                    ),
                    entity_ids=[f"opentargets:disease:{candidate.disease_id}", f"opentargets:drug:{candidate.drug_id}"],
                    title=f"{candidate.drug_name} — {candidate.disease_id}",
                    summary=(
                        f"Open Targets lists {candidate.drug_name} as a disease-level drug candidate; "
                        f"maximum clinical development stage reported: {candidate.max_clinical_stage}. "
                        "This disease-level indication record does not by itself establish a direct gene-drug relationship "
                        "or efficacy for an individual."
                    ),
                )
            )
        return records


class OpenTargetsClient:
    """Small GraphQL client that exposes only target and disease search."""

    def __init__(self, settings: Settings | None = None, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client

    async def search_entities(
        self,
        query: str,
        *,
        entity_type: Literal["target", "disease"],
        limit: int | None = None,
    ) -> OpenTargetsSearchResult:
        """Search one Open Targets entity class; this step does not resolve ambiguity."""

        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 300:
            raise InvalidInputError("Open Targets query must contain 1 to 300 characters", source="opentargets")
        if not isinstance(entity_type, str) or entity_type not in {"target", "disease"}:
            raise InvalidInputError("Open Targets search supports target or disease entities", source="opentargets")
        result_limit = self._settings.max_results_per_source if limit is None else limit
        if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= _MAX_RESULTS:
            raise InvalidInputError(
                f"Open Targets result limit must be between 1 and {_MAX_RESULTS}", source="opentargets"
            )

        variables = {"queryString": query.strip(), "entityNames": [entity_type], "size": result_limit}
        if self._client is None:
            async with create_async_client(self._settings) as client:
                payload = await self._request(client, variables)
        else:
            payload = await self._request(self._client, variables)
        hits, total_count, warnings = self._parse_search(payload, entity_type)
        return OpenTargetsSearchResult(
            query=query.strip(),
            entity_type=entity_type,
            hits=hits,
            total_count=total_count,
            status="empty" if not hits else ("success" if len(hits) == 1 else "ambiguous"),
            warnings=warnings,
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )

    async def get_target_disease_association(
        self,
        target_id: str,
        disease_id: str,
        *,
        limit: int | None = None,
        include_drugs: bool = True,
    ) -> OpenTargetsAssociationResult:
        """Fetch direct associations, optionally omitting disease-level drug candidates."""

        if not isinstance(target_id, str) or not re.fullmatch(r"ENSG[0-9]{11}", target_id):
            raise InvalidInputError("Open Targets target ID must be an Ensembl gene ID", source="opentargets")
        if not isinstance(disease_id, str) or not re.fullmatch(r"(?:EFO|MONDO)_[0-9]{7}", disease_id):
            raise InvalidInputError(
                "Open Targets association query requires a canonical EFO or MONDO disease ID",
                source="opentargets",
            )
        if not isinstance(include_drugs, bool):
            raise InvalidInputError("Open Targets include_drugs must be a boolean", source="opentargets")
        result_limit = self._settings.max_results_per_source if limit is None else limit
        if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= _MAX_RESULTS:
            raise InvalidInputError(
                f"Open Targets result limit must be between 1 and {_MAX_RESULTS}", source="opentargets"
            )

        variables = {"targetId": target_id, "diseaseId": disease_id, "size": result_limit}
        query = _ASSOCIATION_QUERY if include_drugs else _ASSOCIATION_QUERY_NO_DRUGS
        if self._client is None:
            async with create_async_client(self._settings) as client:
                payload = await self._request(client, variables, query=query)
        else:
            payload = await self._request(self._client, variables, query=query)
        return self._parse_associations(
            payload, target_id, disease_id, clinical_limit=result_limit, include_drugs=include_drugs
        )

    async def _request(
        self,
        client: httpx.AsyncClient,
        variables: dict[str, Any],
        *,
        query: str = _SEARCH_QUERY,
    ) -> dict[str, Any]:
        try:
            response = await client.post(
                _GRAPHQL_URL,
                json={"query": query, "variables": variables},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError("Open Targets search request timed out", source="opentargets") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                try:
                    retry_after_seconds = float(retry_after) if retry_after is not None else None
                except ValueError:
                    retry_after_seconds = None
                if retry_after_seconds is not None and retry_after_seconds < 0:
                    retry_after_seconds = None
                raise RateLimitedError(
                    "Open Targets rate limit reached",
                    source="opentargets",
                    retry_after_seconds=retry_after_seconds,
                ) from exc
            raise UpstreamError(
                f"Open Targets search returned HTTP {exc.response.status_code}",
                source="opentargets",
                retryable=exc.response.status_code >= 500,
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamError("Open Targets search request failed", source="opentargets") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("Open Targets returned invalid JSON", source="opentargets", retryable=False) from exc
        if not isinstance(payload, dict):
            raise UpstreamError("Open Targets returned an invalid response", source="opentargets", retryable=False)
        if payload.get("errors"):
            raise UpstreamError("Open Targets GraphQL query failed", source="opentargets", retryable=False)
        return payload

    @staticmethod
    def _parse_associations(
        payload: dict[str, Any], target_id: str, disease_id: str, *, clinical_limit: int, include_drugs: bool = True
    ) -> OpenTargetsAssociationResult:
        data = payload.get("data")
        if not isinstance(data, dict) or "target" not in data:
            raise UpstreamError("Open Targets response is missing data.target", source="opentargets", retryable=False)
        target = data["target"]
        if include_drugs:
            clinical_candidates, clinical_total, warnings = OpenTargetsClient._parse_clinical_candidates(
                data.get("disease"), disease_id, limit=clinical_limit
            )
        else:
            clinical_candidates, clinical_total, warnings = [], 0, []
        if target is None:
            return OpenTargetsAssociationResult(
                target_id=target_id,
                target_symbol=None,
                disease_id=disease_id,
                associations=[],
                clinical_drug_candidates=clinical_candidates,
                clinical_drug_total_count=clinical_total,
                warnings=warnings,
                total_count=0,
                status="target_not_found",
                retrieved_at=datetime.now(timezone.utc),
                request_id=uuid4(),
            )
        if not isinstance(target, dict) or target.get("id") != target_id:
            raise UpstreamError("Open Targets returned an invalid target identity", source="opentargets", retryable=False)
        symbol = target.get("approvedSymbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise UpstreamError("Open Targets target is missing its approved symbol", source="opentargets", retryable=False)
        association_block = target.get("associatedDiseases")
        if not isinstance(association_block, dict):
            raise UpstreamError("Open Targets response is missing associatedDiseases", source="opentargets", retryable=False)
        raw_rows = association_block.get("rows")
        total_count = association_block.get("count")
        if not isinstance(raw_rows, list) or isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
            raise UpstreamError("Open Targets returned invalid association rows", source="opentargets", retryable=False)

        associations: list[OpenTargetsDiseaseAssociation] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                raise UpstreamError("Open Targets returned an invalid association row", source="opentargets", retryable=False)
            disease = raw_row.get("disease")
            if not isinstance(disease, dict) or disease.get("id") != disease_id:
                raise UpstreamError("Open Targets association does not match the requested disease", source="opentargets", retryable=False)
            name = disease.get("name")
            score = raw_row.get("score")
            if not isinstance(name, str) or not name.strip() or isinstance(score, bool) or not isinstance(score, (int, float)):
                raise UpstreamError("Open Targets association is missing its score or disease name", source="opentargets", retryable=False)
            components: dict[str, list[OpenTargetsScoreComponent]] = {}
            for field, output_field in (("datasourceScores", "datasource_scores"), ("datatypeScores", "datatype_scores")):
                raw_components = raw_row.get(field, [])
                if not isinstance(raw_components, list):
                    raise UpstreamError("Open Targets returned invalid score components", source="opentargets", retryable=False)
                parsed: list[OpenTargetsScoreComponent] = []
                for raw_component in raw_components:
                    if not isinstance(raw_component, dict):
                        raise UpstreamError("Open Targets returned an invalid score component", source="opentargets", retryable=False)
                    try:
                        parsed.append(OpenTargetsScoreComponent(identifier=raw_component.get("id"), score=raw_component.get("score")))
                    except (TypeError, ValueError) as exc:
                        raise UpstreamError("Open Targets score component failed schema validation", source="opentargets", retryable=False) from exc
                components[output_field] = parsed
            try:
                associations.append(
                    OpenTargetsDiseaseAssociation(
                        score=score,
                        disease_id=disease_id,
                        disease_name=name.strip(),
                        datasource_scores=components["datasource_scores"],
                        datatype_scores=components["datatype_scores"],
                    )
                )
            except ValueError as exc:
                raise UpstreamError("Open Targets association failed schema validation", source="opentargets", retryable=False) from exc
        if len(associations) > _MAX_RESULTS:
            raise UpstreamError("Open Targets returned more associations than requested", source="opentargets", retryable=False)
        return OpenTargetsAssociationResult(
            target_id=target_id,
            target_symbol=symbol.strip(),
            disease_id=disease_id,
            associations=associations,
            clinical_drug_candidates=clinical_candidates,
            clinical_drug_total_count=clinical_total,
            warnings=warnings,
            total_count=total_count,
            status="success" if associations else "empty",
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )

    @staticmethod
    def _parse_clinical_candidates(
        disease: Any, disease_id: str, *, limit: int
    ) -> tuple[list[OpenTargetsClinicalDrugCandidate], int, list[str]]:
        if disease is None:
            return [], 0, ["disease_record_not_found_for_clinical_candidates"]
        if not isinstance(disease, dict) or disease.get("id") != disease_id:
            raise UpstreamError("Open Targets returned an invalid disease identity", source="opentargets", retryable=False)
        block = disease.get("drugAndClinicalCandidates")
        if not isinstance(block, dict):
            raise UpstreamError("Open Targets response is missing disease clinical candidates", source="opentargets", retryable=False)
        raw_rows = block.get("rows")
        total = block.get("count")
        if not isinstance(raw_rows, list) or isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise UpstreamError("Open Targets returned invalid clinical candidate rows", source="opentargets", retryable=False)

        candidates: list[OpenTargetsClinicalDrugCandidate] = []
        warnings: list[str] = []
        for row in raw_rows[:limit]:
            if not isinstance(row, dict):
                raise UpstreamError("Open Targets returned an invalid clinical candidate", source="opentargets", retryable=False)
            drug = row.get("drug")
            if drug is None:
                warnings.append("clinical_candidate_missing_drug_entity")
                continue
            if not isinstance(drug, dict):
                raise UpstreamError("Open Targets returned an invalid drug entity", source="opentargets", retryable=False)
            try:
                candidates.append(
                    OpenTargetsClinicalDrugCandidate(
                        indication_id=row.get("id"),
                        disease_id=disease_id,
                        drug_id=drug.get("id"),
                        drug_name=drug.get("name"),
                        max_clinical_stage=row.get("maxClinicalStage"),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise UpstreamError("Open Targets clinical candidate failed schema validation", source="opentargets", retryable=False) from exc
        if total > limit or len(raw_rows) > limit:
            warnings.append("clinical_drug_candidates_truncated_by_result_limit")
        return candidates, total, list(dict.fromkeys(warnings))

    @staticmethod
    def _parse_search(
        payload: dict[str, Any], entity_type: Literal["target", "disease"]
    ) -> tuple[list[OpenTargetsSearchHit], int, list[str]]:
        data = payload.get("data")
        search = data.get("search") if isinstance(data, dict) else None
        if not isinstance(search, dict):
            raise UpstreamError("Open Targets response is missing data.search", source="opentargets", retryable=False)
        raw_hits = search.get("hits")
        total = search.get("total")
        if not isinstance(raw_hits, list) or isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise UpstreamError("Open Targets response contains invalid search results", source="opentargets", retryable=False)

        hits: list[OpenTargetsSearchHit] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, dict):
                raise UpstreamError("Open Targets response contains an invalid hit", source="opentargets", retryable=False)
            returned_type = raw_hit.get("entity")
            identifier = raw_hit.get("id")
            name = raw_hit.get("name")
            if returned_type != entity_type or not isinstance(identifier, str) or not identifier.strip():
                raise UpstreamError("Open Targets hit has an unexpected entity type or missing ID", source="opentargets", retryable=False)
            if not isinstance(name, str) or not name.strip():
                raise UpstreamError("Open Targets hit is missing its display name", source="opentargets", retryable=False)
            if identifier in seen_ids:
                warnings.append("duplicate_entity_hit_removed")
                continue
            seen_ids.add(identifier)
            description = raw_hit.get("description")
            score = raw_hit.get("score")
            if description is not None and not isinstance(description, str):
                description = None
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
                score = None
            try:
                hits.append(
                    OpenTargetsSearchHit(
                        entity_type=entity_type,
                        identifier=identifier.strip(),
                        name=name.strip(),
                        description=description.strip() if description and description.strip() else None,
                        score=score,
                    )
                )
            except ValueError as exc:
                raise UpstreamError("Open Targets hit failed schema validation", source="opentargets", retryable=False) from exc
        if len(hits) > _MAX_RESULTS:
            raise UpstreamError("Open Targets returned more hits than requested", source="opentargets", retryable=False)
        return hits, total, list(dict.fromkeys(warnings))
