"""Human gene-name search adapter for the UniProtKB REST API."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import Field, field_validator, model_validator
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from gene_evidence_agent.cache import TTLCache
from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import InvalidInputError, RateLimitedError, ToolTimeoutError, UpstreamError
from gene_evidence_agent.http_client import create_async_client
from gene_evidence_agent.rate_limit import AsyncRateLimiter
from gene_evidence_agent.schemas import BiomedicalEntity, StrictModel

_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_HUMAN_TAXON_ID = 9606
_MAX_RESULTS = 50
_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 5.0
_BACKOFF = wait_exponential_jitter(initial=0.1, max=0.5)
_GENE_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class UniProtEntryCandidate(StrictModel):
    """Candidate identity and source-provided annotations from UniProtKB."""

    accession: str = Field(min_length=1, max_length=20)
    entry_name: str | None = Field(default=None, max_length=100)
    protein_name: str | None = Field(default=None, max_length=500)
    gene_names: list[str] = Field(default_factory=list, max_length=100)
    function_annotations: list[str] = Field(default_factory=list, max_length=50)
    disease_annotations: list[str] = Field(default_factory=list, max_length=50)
    organism_name: str = Field(min_length=1, max_length=200)
    organism_tax_id: int = Field(ge=1)
    reviewed: bool
    record_url: str = Field(min_length=1, max_length=300)


class UniProtSearchResult(StrictModel):
    """Validated result envelope for one exact human gene-name query."""

    query: str = Field(min_length=1, max_length=300)
    gene_symbol: str = Field(min_length=1, max_length=64)
    entries: list[UniProtEntryCandidate] = Field(default_factory=list, max_length=_MAX_RESULTS)
    status: Literal["success", "empty", "ambiguous"]
    selected_accession: str | None = Field(default=None, max_length=20)
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
    def validate_status(self) -> UniProtSearchResult:
        if len(self.entries) > 1:
            expected_status = "ambiguous"
        elif self.entries:
            expected_status = "success"
        else:
            expected_status = "empty"
        if self.status != expected_status:
            raise ValueError(f"status must be {expected_status!r} for this entry list")
        if expected_status == "success" and self.selected_accession != self.entries[0].accession:
            raise ValueError("a single result must select its only accession")
        if expected_status != "success" and self.selected_accession is not None:
            raise ValueError("empty or ambiguous results must not select an accession")
        return self


class UniProtClient:
    """Bounded UniProtKB search client with process-local pacing and caching."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: TTLCache[str, UniProtSearchResult] | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._cache = cache or TTLCache(ttl_seconds=self._settings.cache_ttl_seconds)
        self._rate_limiter = rate_limiter or AsyncRateLimiter(0.34)

    async def search_human_gene(
        self,
        gene: BiomedicalEntity | str,
        *,
        limit: int | None = None,
    ) -> UniProtSearchResult:
        """Search UniProtKB by an exact gene symbol and human taxonomy ID."""

        gene_symbol = _normalize_gene_symbol(gene)
        result_limit = self._settings.max_results_per_source if limit is None else limit
        if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= _MAX_RESULTS:
            raise InvalidInputError(
                f"UniProt result limit must be between 1 and {_MAX_RESULTS}",
                source="uniprot",
            )
        query = f"gene_exact:{gene_symbol} AND organism_id:{_HUMAN_TAXON_ID}"
        params = {"query": query, "format": "json", "size": result_limit}
        cache_key = _cache_key(gene_symbol, result_limit)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        if self._client is None:
            async with create_async_client(self._settings) as client:
                payload = await self._request(client, params)
        else:
            payload = await self._request(self._client, params)

        entries = prioritize_candidates(self._parse_entries(payload, gene_symbol))
        if len(entries) > 1:
            status: Literal["success", "empty", "ambiguous"] = "ambiguous"
            selected_accession = None
            warnings = ["multiple_exact_gene_entries_require_disambiguation"]
        elif entries:
            status = "success"
            selected_accession = entries[0].accession
            warnings = [] if entries[0].reviewed else ["only_unreviewed_candidate"]
        else:
            status = "empty"
            selected_accession = None
            warnings = []
        result = UniProtSearchResult(
            query=query,
            gene_symbol=gene_symbol,
            entries=entries,
            status=status,
            selected_accession=selected_accession,
            warnings=warnings,
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )
        self._cache.set(cache_key, result)
        return result

    async def _request(self, client: httpx.AsyncClient, params: dict[str, str | int]) -> dict[str, Any]:
        response = await self._get_response_with_retry(client, params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("UniProt returned invalid JSON", source="uniprot", retryable=False) from exc
        if not isinstance(payload, dict):
            raise UpstreamError("UniProt returned an invalid response", source="uniprot", retryable=False)
        return payload

    async def _get_response_with_retry(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=_retry_wait,
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
        ):
            with attempt:
                return await self._get_response(client, params)
        raise AssertionError("Retrying exited without a response or exception")

    async def _get_response(self, client: httpx.AsyncClient, params: dict[str, str | int]) -> httpx.Response:
        await self._rate_limiter.acquire()
        try:
            response = await client.get(_SEARCH_URL, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError("UniProt search request timed out", source="uniprot") from exc
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
                    "UniProt rate limit reached",
                    source="uniprot",
                    retry_after_seconds=retry_after_seconds,
                ) from exc
            raise UpstreamError(
                f"UniProt search returned HTTP {exc.response.status_code}",
                source="uniprot",
                retryable=exc.response.status_code >= 500,
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamError("UniProt search request failed", source="uniprot") from exc
        return response

    @staticmethod
    def _parse_entries(payload: dict[str, Any], gene_symbol: str) -> list[UniProtEntryCandidate]:
        raw_entries = payload.get("results")
        if not isinstance(raw_entries, list):
            raise UpstreamError("UniProt response is missing its results list", source="uniprot")

        entries: list[UniProtEntryCandidate] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise UpstreamError("UniProt response contains an invalid entry", source="uniprot")
            candidate = _parse_entry(raw_entry)
            if candidate.organism_tax_id != _HUMAN_TAXON_ID:
                raise UpstreamError("UniProt returned a non-human entry for a human-only query", source="uniprot")
            if gene_symbol.casefold() not in {name.casefold() for name in candidate.gene_names}:
                raise UpstreamError("UniProt returned an entry without the exact queried gene symbol", source="uniprot")
            entries.append(candidate)
        return entries


def _normalize_gene_symbol(gene: BiomedicalEntity | str) -> str:
    if isinstance(gene, BiomedicalEntity):
        if gene.entity_type != "gene":
            raise InvalidInputError("UniProt search requires a gene entity", source="uniprot")
        if gene.organism_tax_id not in (None, _HUMAN_TAXON_ID):
            raise InvalidInputError("UniProt MVP search is limited to human genes", source="uniprot")
        gene_symbol = gene.query_name.strip()
    elif isinstance(gene, str):
        gene_symbol = gene.strip()
    else:
        raise InvalidInputError("Gene symbol must be text or a biomedical gene entity", source="uniprot")

    if not _GENE_SYMBOL_PATTERN.fullmatch(gene_symbol):
        raise InvalidInputError("Gene symbol contains unsupported query syntax", source="uniprot")
    return gene_symbol


def prioritize_candidates(
    entries: list[UniProtEntryCandidate],
) -> list[UniProtEntryCandidate]:
    """Put reviewed entries first and remove duplicate accessions stably."""

    ranked = sorted(entries, key=lambda candidate: not candidate.reviewed)
    unique: list[UniProtEntryCandidate] = []
    seen_accessions: set[str] = set()
    for candidate in ranked:
        if candidate.accession not in seen_accessions:
            unique.append(candidate)
            seen_accessions.add(candidate.accession)
    return unique


def _parse_entry(raw_entry: dict[str, Any]) -> UniProtEntryCandidate:
    accession = raw_entry.get("primaryAccession")
    entry_name = raw_entry.get("uniProtkbId")
    entry_type = raw_entry.get("entryType", "")
    organism = raw_entry.get("organism")
    protein_description = raw_entry.get("proteinDescription")
    if not isinstance(accession, str) or not isinstance(organism, dict):
        raise UpstreamError("UniProt entry is missing required identity fields", source="uniprot")

    taxon_id = organism.get("taxonId")
    organism_name = organism.get("scientificName")
    if not isinstance(taxon_id, int) or not isinstance(organism_name, str) or not organism_name.strip():
        raise UpstreamError("UniProt entry contains invalid organism data", source="uniprot")

    protein_name: str | None = None
    if isinstance(protein_description, dict):
        recommended = protein_description.get("recommendedName")
        if isinstance(recommended, dict):
            full_name = recommended.get("fullName")
            if isinstance(full_name, dict) and isinstance(full_name.get("value"), str):
                protein_name = full_name["value"]

    genes = raw_entry.get("genes", [])
    if not isinstance(genes, list):
        raise UpstreamError("UniProt entry contains invalid gene-name data", source="uniprot")
    gene_names: list[str] = []
    for gene in genes:
        if not isinstance(gene, dict):
            continue
        for field in ("geneName", "synonyms", "orderedLocusNames", "orfNames"):
            raw_names = gene.get(field)
            values = raw_names if isinstance(raw_names, list) else [raw_names]
            for item in values:
                if isinstance(item, dict) and isinstance(item.get("value"), str):
                    value = item["value"].strip()
                    if value and value.casefold() not in {name.casefold() for name in gene_names}:
                        gene_names.append(value)

    function_annotations, disease_annotations = _parse_annotations(raw_entry.get("comments", []))

    normalized_entry_type = entry_type.strip().casefold() if isinstance(entry_type, str) else ""
    reviewed = normalized_entry_type.startswith("uniprotkb reviewed ")
    try:
        return UniProtEntryCandidate(
            accession=accession,
            entry_name=entry_name if isinstance(entry_name, str) else None,
            protein_name=protein_name,
            gene_names=gene_names,
            function_annotations=function_annotations,
            disease_annotations=disease_annotations,
            organism_name=organism_name,
            organism_tax_id=taxon_id,
            reviewed=reviewed,
            record_url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
        )
    except ValueError as exc:
        raise UpstreamError("UniProt entry failed schema validation", source="uniprot") from exc


def _parse_annotations(comments: Any) -> tuple[list[str], list[str]]:
    """Extract verbatim FUNCTION and DISEASE annotation text from UniProt comments."""

    if not isinstance(comments, list):
        raise UpstreamError("UniProt entry contains invalid annotation data", source="uniprot")

    functions: list[str] = []
    diseases: list[str] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        comment_type = comment.get("commentType")
        if not isinstance(comment_type, str) or comment_type not in {"FUNCTION", "DISEASE"}:
            continue

        texts: list[str] = []
        raw_texts = comment.get("texts", [])
        if isinstance(raw_texts, list):
            texts.extend(
                value.strip()
                for item in raw_texts
                if isinstance(item, dict)
                and isinstance((value := item.get("value")), str)
                and value.strip()
            )

        if comment_type == "DISEASE":
            disease = comment.get("disease")
            label = ""
            if isinstance(disease, dict):
                identifiers = [
                    value.strip()
                    for key in ("acronym", "diseaseId")
                    if isinstance((value := disease.get(key)), str) and value.strip()
                ]
                label = " (" + ", ".join(identifiers) + ")" if identifiers else ""
                description = disease.get("description")
                if isinstance(description, str) and description.strip():
                    texts.insert(0, description.strip())
            note = comment.get("note")
            if isinstance(note, dict) and isinstance(note.get("texts"), list):
                texts.extend(
                    value.strip()
                    for item in note["texts"]
                    if isinstance(item, dict)
                    and isinstance((value := item.get("value")), str)
                    and value.strip()
                )
            target = diseases
        else:
            label = ""
            target = functions

        for value in texts:
            annotation = f"{label}: {value}" if label else value
            if len(annotation) > 4000:
                raise UpstreamError("UniProt annotation exceeds the supported size", source="uniprot")
            if annotation.casefold() not in {existing.casefold() for existing in target}:
                target.append(annotation)
            if len(target) > 50:
                raise UpstreamError("UniProt returned too many annotations", source="uniprot")

    return functions, diseases


def _cache_key(gene_symbol: str, result_limit: int) -> str:
    digest = sha256(f"{gene_symbol.casefold()}\0{result_limit}".encode("utf-8")).hexdigest()
    return f"uniprot:{digest}"


def _retry_wait(retry_state: RetryCallState) -> float:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exception, RateLimitedError) and exception.retry_after_seconds is not None:
        return min(exception.retry_after_seconds, _MAX_RETRY_AFTER_SECONDS)
    return _BACKOFF(retry_state)


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, RateLimitedError):
        return error.retry_after_seconds is None or error.retry_after_seconds <= _MAX_RETRY_AFTER_SECONDS
    if isinstance(error, ToolTimeoutError):
        return True
    if isinstance(error, UpstreamError):
        return error.retryable
    return False
