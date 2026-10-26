"""Conservative preflight checks that keep obvious patient identifiers out of APIs."""

from __future__ import annotations

import re

from gene_evidence_agent.errors import InvalidInputError

_DIRECT_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email_address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("phone_number", re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d{9}(?!\d)")),
    ("government_id", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\w)|\b\d{3}-\d{2}-\d{4}\b")),
    (
        "medical_record_id",
        re.compile(
            r"(?:病历号|住院号|患者(?:ID|编号)|patient\s*(?:id|number)|medical\s*record\s*(?:no|number)|\bMRN)"
            r"\s*(?::|：|#|\bis\b)?\s*[A-Z0-9][A-Z0-9-]{3,}",
            re.IGNORECASE,
        ),
    ),
)

_PATIENT_CASE_PATTERN = re.compile(
    r"\b(?:my|our|this)\s+patient\b|\bpatient(?:'s|\s+(?:case|record|data))\b|"
    r"(?:患者|病人)(?:的|病例|信息|資料|资料|数据|病历)",
    re.IGNORECASE,
)

_PERSONAL_MEDICAL_PATTERN = re.compile(
    r"\b(?:for me|my\s+(?:treatment|diagnosis|prognosis|medication|dose)|"
    r"should\s+i\s+(?:take|use|start|stop)|what\s+(?:should|can)\s+i\s+take)\b|"
    r"(?:给我(?:用药|治疗)|我(?:应该|能否|是否).{0,12}(?:用药|治疗)|"
    r"我的(?:治疗|诊断|预后|药物|剂量))",
    re.IGNORECASE,
)


def sensitive_query_category(query: str) -> str | None:
    """Return only a category code; never return or retain matched user text."""

    if _PATIENT_CASE_PATTERN.search(query):
        return "patient_specific_context"
    if _PERSONAL_MEDICAL_PATTERN.search(query):
        return "personal_medical_advice"
    for category, pattern in _DIRECT_IDENTIFIER_PATTERNS:
        if pattern.search(query):
            return category
    return None


def ensure_no_sensitive_query(query: str) -> None:
    """Reject likely patient/personal medical data without echoing matched text."""

    if sensitive_query_category(query) is not None:
        raise InvalidInputError(
            "This research evidence tool cannot handle patient-specific details or personal medical advice; "
            "remove identifiers and use a population-level research question",
            source="privacy",
        )
