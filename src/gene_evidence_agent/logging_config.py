"""Application logging setup with no credential or payload logging."""

from __future__ import annotations

import logging
import re

from gene_evidence_agent.config import get_settings

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ".,);]"


class RedactURLQueryFilter(logging.Filter):
    """Remove URL queries/fragments from logs to avoid leaking keys or queries."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            record.msg = "Log message redacted because it could not be formatted safely"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            return True

        record.msg = _URL_PATTERN.sub(_redact_url_query, message)
        record.args = ()
        if record.name.startswith("httpx") and record.exc_info:
            # HTTP client exception text can include the complete request URL.
            record.exc_info = None
            record.exc_text = None
        return True


def _redact_url_query(match: re.Match[str]) -> str:
    url = match.group(0)
    trailing = ""
    while url and url[-1] in _URL_TRAILING_PUNCTUATION:
        trailing = url[-1] + trailing
        url = url[:-1]
    cut_points = [position for marker in ("?", "#") if (position := url.find(marker)) >= 0]
    if not cut_points:
        return url + trailing
    return url[: min(cut_points)] + "?[REDACTED]" + trailing


def configure_logging(level: str | int | None = None) -> None:
    """Configure standard logging once, using the configured level by default."""

    selected_level: str | int = level if level is not None else get_settings().log_level
    if isinstance(selected_level, str):
        normalized = selected_level.strip().upper()
        numeric_level = logging.getLevelName(normalized)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Unsupported log level: {selected_level}")
        selected_level = numeric_level

    logging.basicConfig(
        level=selected_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(item, RedactURLQueryFilter) for item in handler.filters):
            handler.addFilter(RedactURLQueryFilter())

