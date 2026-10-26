"""Conservative Chinese-to-English normalization for disease search terms."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


class UnsupportedChineseDiseaseTerm(ValueError):
    """Raised when a Chinese disease term has no exact supported mapping."""


@dataclass(frozen=True)
class DiseaseSearchTerm:
    original: str
    search_term: str
    translated: bool


_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def contains_chinese_text(value: str) -> bool:
    """Whether a query contains CJK ideographs that sources may not parse."""

    return bool(_CHINESE_RE.search(value))


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    # Keep parenthetical acronyms: they can distinguish a cancer subtype.
    return re.sub(r"[\s\W_]+", "", normalized).casefold()


def _aliases(canonical: str, *names: str) -> tuple[str, tuple[str, ...]]:
    return canonical, names


_CANCER_TERMS = (
    _aliases("lung cancer", "肺癌", "肺部癌", "肺部癌症"),
    _aliases("non-small cell lung cancer", "非小细胞肺癌", "非小细胞肺癌症", "肺癌NSCLC", "非小细胞肺癌NSCLC"),
    _aliases("small cell lung cancer", "小细胞肺癌", "小细胞肺癌症", "肺癌SCLC", "小细胞肺癌SCLC"),
    _aliases("lung adenocarcinoma", "肺腺癌", "肺腺细胞癌"),
    _aliases("lung squamous cell carcinoma", "肺鳞癌", "肺鳞状细胞癌"),
    _aliases("breast cancer", "乳腺癌", "乳癌"),
    _aliases("triple-negative breast cancer", "三阴性乳腺癌", "三阴性乳癌", "乳腺癌TNBC"),
    _aliases("colorectal cancer", "结直肠癌", "大肠癌"),
    _aliases("colon cancer", "结肠癌"),
    _aliases("rectal cancer", "直肠癌"),
    _aliases("liver cancer", "肝癌"),
    _aliases("hepatocellular carcinoma", "肝细胞癌"),
    _aliases("gastric cancer", "胃癌"),
    _aliases("esophageal cancer", "食管癌", "食道癌"),
    _aliases("pancreatic cancer", "胰腺癌"),
    _aliases("prostate cancer", "前列腺癌"),
    _aliases("ovarian cancer", "卵巢癌"),
    _aliases("cervical cancer", "宫颈癌", "子宫颈癌"),
    _aliases("endometrial cancer", "子宫内膜癌"),
    _aliases("bladder cancer", "膀胱癌"),
    _aliases("renal cell carcinoma", "肾细胞癌"),
    _aliases("kidney cancer", "肾癌"),
    _aliases("thyroid cancer", "甲状腺癌"),
    _aliases("glioblastoma", "胶质母细胞瘤"),
    _aliases("glioma", "胶质瘤"),
    _aliases("brain cancer", "脑癌", "脑部癌症"),
    _aliases("melanoma", "黑色素瘤"),
    _aliases("leukemia", "白血病", "白血癌"),
    _aliases("acute myeloid leukemia", "急性髓系白血病", "急性髓细胞白血病"),
    _aliases("acute lymphoblastic leukemia", "急性淋巴细胞白血病"),
    _aliases("chronic lymphocytic leukemia", "慢性淋巴细胞白血病"),
    _aliases("chronic myeloid leukemia", "慢性髓系白血病", "慢性髓细胞白血病"),
    _aliases("lymphoma", "淋巴瘤"),
    _aliases("Hodgkin lymphoma", "霍奇金淋巴瘤"),
    _aliases("non-Hodgkin lymphoma", "非霍奇金淋巴瘤"),
    _aliases("multiple myeloma", "多发性骨髓瘤"),
    _aliases("sarcoma", "肉瘤"),
    _aliases("osteosarcoma", "骨肉瘤"),
    _aliases("nasopharyngeal carcinoma", "鼻咽癌"),
    _aliases("cholangiocarcinoma", "胆管癌"),
    _aliases("gallbladder cancer", "胆囊癌"),
    _aliases("oral cancer", "口腔癌"),
    _aliases("head and neck cancer", "头颈癌", "头颈部癌"),
    _aliases("laryngeal cancer", "喉癌"),
    _aliases("testicular cancer", "睾丸癌"),
    _aliases("skin cancer", "皮肤癌"),
)

_CHINESE_TO_ENGLISH = {
    _key(alias): canonical
    for canonical, aliases in _CANCER_TERMS
    for alias in aliases
}


def normalize_disease_search_term(value: str) -> DiseaseSearchTerm:
    """Return an English term for supported Chinese cancer names.

    English input is passed through unchanged. Chinese input must exactly match
    a curated alias apart from spacing or punctuation; subtype acronyms are kept
    as part of the match. Unknown terms fail closed so Chinese disease text is
    never sent to a source.
    """

    original = value.strip()
    if not original:
        raise ValueError("Disease search term must not be blank")
    if not contains_chinese_text(original):
        return DiseaseSearchTerm(original=original, search_term=original, translated=False)
    mapped = _CHINESE_TO_ENGLISH.get(_key(original))
    if mapped is None:
        raise UnsupportedChineseDiseaseTerm(original)
    return DiseaseSearchTerm(original=original, search_term=mapped, translated=True)
