"""NCBI E-utilities client for PubMed PMID searches."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4
from xml.etree import ElementTree

import httpx
from pydantic import Field, field_validator, model_validator
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from gene_evidence_agent.cache import TTLCache
from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import (
    InvalidInputError,
    RateLimitedError,
    ToolError,
    ToolTimeoutError,
    UpstreamError,
)
from gene_evidence_agent.http_client import create_async_client
from gene_evidence_agent.rate_limit import AsyncRateLimiter
from gene_evidence_agent.schemas import EvidenceRecord, SourceReference, StrictModel

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_MAX_RESULTS = 50
_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 5.0
_BACKOFF = wait_exponential_jitter(initial=0.1, max=1.0)


class PubMedSearchResult(StrictModel):
    """Validated PMID list and provenance for one ESearch request."""

    query: str = Field(min_length=1, max_length=5000)
    pmids: list[str] = Field(default_factory=list, max_length=_MAX_RESULTS)
    total_count: int = Field(ge=0)
    status: Literal["success", "empty"]
    retrieved_at: datetime
    request_id: UUID

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("pmids")
    @classmethod
    def require_numeric_distinct_pmids(cls, value: list[str]) -> list[str]:
        if any(not pmid.isdigit() for pmid in value):
            raise ValueError("PMIDs must contain digits only")
        if len(value) != len(set(value)):
            raise ValueError("PMIDs must be unique")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> PubMedSearchResult:
        expected_status = "success" if self.pmids else "empty"
        if self.status != expected_status:
            raise ValueError(f"status must be {expected_status!r} for this PMID list")
        if self.total_count < len(self.pmids):
            raise ValueError("total_count cannot be smaller than the returned PMID count")
        return self


class PubMedFetchResult(StrictModel):
    """Batched EFetch XML response kept intact for deterministic parsing."""

    requested_pmids: list[str] = Field(min_length=1, max_length=_MAX_RESULTS)
    response_xml: str = Field(min_length=1, max_length=5_000_000)
    retrieved_at: datetime
    request_id: UUID

    @field_validator("requested_pmids")
    @classmethod
    def require_numeric_distinct_pmids(cls, value: list[str]) -> list[str]:
        if any(not pmid.isdigit() for pmid in value):
            raise ValueError("PMIDs must contain digits only")
        if len(value) != len(set(value)):
            raise ValueError("requested PMIDs must be unique")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value


class PubMedClient:
    """Small E-utilities adapter; retries and caching are added in P3.5."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: TTLCache[str, PubMedSearchResult | PubMedFetchResult] | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._cache = cache or TTLCache(ttl_seconds=self._settings.cache_ttl_seconds)
        requests_per_second = 10 if self._settings.ncbi_api_key else 3
        self._rate_limiter = rate_limiter or AsyncRateLimiter(1 / requests_per_second)

    async def search(self, query: str, *, limit: int | None = None) -> PubMedSearchResult:
        """Run one bounded ESearch query and return validated PubMed IDs."""

        if not isinstance(query, str) or not query.strip() or len(query) > 5000:
            raise InvalidInputError(
                "PubMed query must contain 1 to 5000 characters",
                source="pubmed",
            )
        query = query.strip()
        result_limit = self._settings.max_results_per_source if limit is None else limit
        if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= _MAX_RESULTS:
            raise InvalidInputError(
                f"PubMed result limit must be between 1 and {_MAX_RESULTS}",
                source="pubmed",
            )

        cache_key = _cache_key("esearch", query, str(result_limit))
        cached_result = self._cache.get(cache_key)
        if isinstance(cached_result, PubMedSearchResult):
            return cached_result

        params: dict[str, str | int] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": result_limit,
            "sort": "relevance",
            "tool": "gene-evidence-agent",
        }
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key.get_secret_value()

        if self._client is None:
            async with create_async_client(self._settings) as client:
                payload = await self._request(client, _ESEARCH_URL, params)
        else:
            payload = await self._request(self._client, _ESEARCH_URL, params)

        pmids, total_count = self._parse_response(payload)
        result = PubMedSearchResult(
            query=query,
            pmids=pmids,
            total_count=total_count,
            status="success" if pmids else "empty",
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )
        self._cache.set(cache_key, result)
        return result

    async def _request(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
    ) -> dict[str, Any]:
        response = await self._get_response_with_retry(client, url, params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("PubMed ESearch returned invalid JSON", source="pubmed") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("PubMed ESearch returned an invalid response", source="pubmed")
        return payload

    async def _get_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
        operation: Literal["ESearch", "EFetch"] = "ESearch",
    ) -> httpx.Response:
        if not self._settings.ncbi_email:
            raise InvalidInputError(
                "Set GEA_NCBI_EMAIL to the software developer's contact email before PubMed requests",
                source="pubmed",
            )
        await self._rate_limiter.acquire()
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(f"PubMed {operation} request timed out", source="pubmed") from exc
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
                    "PubMed rate limit reached",
                    source="pubmed",
                    retry_after_seconds=retry_after_seconds,
                ) from exc
            raise UpstreamError(
                f"PubMed {operation} returned HTTP {exc.response.status_code}",
                source="pubmed",
                retryable=exc.response.status_code >= 500,
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamError(f"PubMed {operation} request failed", source="pubmed") from exc
        return response

    async def _get_response_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
        operation: Literal["ESearch", "EFetch"] = "ESearch",
    ) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=_retry_wait,
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
        ):
            with attempt:
                return await self._get_response(client, url, params, operation)
        raise AssertionError("Retrying exited without a response or exception")

    async def fetch_xml(self, pmids: list[str]) -> PubMedFetchResult:
        """Fetch a bounded batch of PubMed records as validated XML."""

        if not 1 <= len(pmids) <= _MAX_RESULTS:
            raise InvalidInputError(
                f"PubMed fetch requires between 1 and {_MAX_RESULTS} PMIDs",
                source="pubmed",
            )
        if any(not isinstance(pmid, str) or not pmid.isdigit() for pmid in pmids):
            raise InvalidInputError("PubMed IDs must contain digits only", source="pubmed")
        if len(pmids) != len(set(pmids)):
            raise InvalidInputError("PubMed IDs must be unique", source="pubmed")

        cache_key = _cache_key("efetch", *pmids)
        cached_result = self._cache.get(cache_key)
        if isinstance(cached_result, PubMedFetchResult):
            return cached_result

        params: dict[str, str | int] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "gene-evidence-agent",
        }
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key.get_secret_value()

        if self._client is None:
            async with create_async_client(self._settings) as client:
                response = await self._get_response_with_retry(client, _EFETCH_URL, params, "EFetch")
        else:
            response = await self._get_response_with_retry(self._client, _EFETCH_URL, params, "EFetch")

        response_xml = response.text
        if not response_xml or len(response_xml) > 5_000_000:
            raise UpstreamError("PubMed EFetch returned an empty or oversized XML response", source="pubmed")
        try:
            root = ElementTree.fromstring(response_xml)
        except ElementTree.ParseError as exc:
            raise UpstreamError("PubMed EFetch returned malformed XML", source="pubmed") from exc
        if root.tag != "PubmedArticleSet":
            raise UpstreamError("PubMed EFetch response has an unexpected XML root", source="pubmed")

        result = PubMedFetchResult(
            requested_pmids=pmids,
            response_xml=response_xml,
            retrieved_at=datetime.now(timezone.utc),
            request_id=uuid4(),
        )
        self._cache.set(cache_key, result)
        return result

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> tuple[list[str], int]:
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            raise UpstreamError("PubMed ESearch response is missing esearchresult", source="pubmed")

        raw_count = result.get("count")
        raw_ids = result.get("idlist")
        if not isinstance(raw_count, str) or not raw_count.isdigit():
            raise UpstreamError("PubMed ESearch response contains an invalid count", source="pubmed")
        if not isinstance(raw_ids, list) or any(
            not isinstance(pmid, str) or not pmid.isdigit() for pmid in raw_ids
        ):
            raise UpstreamError("PubMed ESearch response contains invalid PMIDs", source="pubmed")

        pmids = list(dict.fromkeys(raw_ids))
        if len(pmids) > _MAX_RESULTS:
            raise UpstreamError("PubMed ESearch returned more IDs than the configured maximum", source="pubmed")
        total_count = int(raw_count)
        if total_count < len(pmids):
            raise UpstreamError("PubMed ESearch count is smaller than its returned PMID list", source="pubmed")
        return pmids, total_count


def parse_pubmed_evidence(
    fetch_result: PubMedFetchResult,
    *,
    entity_ids: list[str],
) -> list[EvidenceRecord]:
    """Convert EFetch XML records into source-linked evidence objects."""

    if not entity_ids or any(not entity_id.strip() for entity_id in entity_ids):
        raise InvalidInputError("At least one non-blank entity ID is required", source="pubmed")
    if len(entity_ids) != len(set(entity_ids)):
        raise InvalidInputError("Entity IDs must be unique", source="pubmed")

    try:
        root = ElementTree.fromstring(fetch_result.response_xml)
    except ElementTree.ParseError as exc:
        raise UpstreamError("PubMed EFetch XML could not be parsed", source="pubmed") from exc
    if root.tag != "PubmedArticleSet":
        raise UpstreamError("PubMed EFetch response has an unexpected XML root", source="pubmed")

    requested_pmids = set(fetch_result.requested_pmids)
    records: list[EvidenceRecord] = []
    seen_pmids: set[str] = set()
    for article in root.findall("PubmedArticle"):
        pmid = _element_text(article.find("./MedlineCitation/PMID"))
        if pmid is None or pmid not in requested_pmids:
            raise UpstreamError("PubMed EFetch returned a record with an unexpected PMID", source="pubmed")
        if pmid in seen_pmids:
            continue
        seen_pmids.add(pmid)

        article_node = article.find("./MedlineCitation/Article")
        title = _element_text(article_node.find("ArticleTitle")) if article_node is not None else None
        journal = (
            _element_text(article_node.find("./Journal/Title"))
            if article_node is not None
            else None
        )
        abstract = _extract_abstract(article_node) if article_node is not None else None
        summary = abstract or title or "PubMed did not supply a title or abstract; consult the source record."
        warnings: list[str] = []
        if not abstract:
            warnings.append("abstract_not_available")
        if not title:
            warnings.append("title_not_available")

        records.append(
            EvidenceRecord(
                evidence_type="publication",
                source=SourceReference(
                    source_database="PubMed",
                    record_id=pmid,
                    record_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    retrieved_at=fetch_result.retrieved_at,
                ),
                entity_ids=entity_ids,
                title=title,
                summary=summary,
                journal=journal,
                publication_date=_extract_publication_date(article_node),
                doi=_extract_doi(article),
                pmid=pmid,
                publication_types=list(dict.fromkeys(value for node in article_node.findall('./PublicationTypeList/PublicationType')
                    if (value:=_element_text(node)))) if article_node is not None else [],
                publication_languages=list(dict.fromkeys(value.casefold() for node in article_node.findall('./Language')
                    if (value:=_element_text(node)))) if article_node is not None else [],
                warnings=warnings,
            )
        )

    return records


def _element_text(element: ElementTree.Element | None) -> str | None:
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _extract_abstract(article_node: ElementTree.Element) -> str | None:
    abstract_sections = article_node.findall("./Abstract/AbstractText")
    parts: list[str] = []
    for section in abstract_sections:
        text = _element_text(section)
        if text:
            label = section.attrib.get("Label")
            parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts) or None


def _extract_publication_date(article_node: ElementTree.Element) -> str | None:
    date_nodes = [
        article_node.find("./Journal/JournalIssue/PubDate"),
        article_node.find("./ArticleDate"),
    ]
    for date_node in date_nodes:
        if date_node is None:
            continue
        medline_date = _element_text(date_node.find("MedlineDate"))
        if medline_date:
            return medline_date
        parts = [
            _element_text(date_node.find(field))
            for field in ("Year", "Month", "Day")
        ]
        date_text = " ".join(part for part in parts if part)
        if date_text:
            return date_text
    return None


def _extract_doi(article: ElementTree.Element) -> str | None:
    for location in article.findall(".//ELocationID"):
        if location.attrib.get("EIdType", "").casefold() == "doi":
            doi = _element_text(location)
            if doi:
                return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    for article_id in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType", "").casefold() == "doi":
            doi = _element_text(article_id)
            if doi:
                return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return None


def _cache_key(namespace: str, *parts: str) -> str:
    digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _retry_wait(retry_state: RetryCallState) -> float:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exception, RateLimitedError) and exception.retry_after_seconds is not None:
        return min(exception.retry_after_seconds, _MAX_RETRY_AFTER_SECONDS)
    return _BACKOFF(retry_state)


def _is_retryable_error(error: BaseException) -> bool:
    if not isinstance(error, ToolError) or not error.retryable:
        return False
    if (
        isinstance(error, RateLimitedError)
        and error.retry_after_seconds is not None
        and error.retry_after_seconds > _MAX_RETRY_AFTER_SECONDS
    ):
        return False
    return True
