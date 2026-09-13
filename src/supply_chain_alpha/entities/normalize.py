from __future__ import annotations

import re
import unicodedata
from enum import Enum

_PUNCT = re.compile(r"[\s\u3000·•,，.。()（）\[\]【】{}<>《》'\"“”‘’_-]+")
_DASH_ONLY = re.compile(r"[-\u2010-\u2015\u2212]+")
_STAR_ONLY = re.compile(r"[*]+")
_MISSING_NAME_TOKENS = frozenset(
    {
        "n/a",
        "na",
        "none",
        "unknown",
        "不详",
        "不适用",
        "无",
        "未提供",
        "未披露",
        "未知",
    }
)
_MISSING_TOKEN_ALTERNATIVES = "|".join(
    re.escape(token)
    for token in sorted(_MISSING_NAME_TOKENS, key=lambda token: (-len(token), token))
)
_MISSING_TOKEN_SEPARATOR = r"[\s\u3000,，.。:：;；/\\|、()（）\[\]【】'\"“”‘’_-]+"
_MISSING_NAME = re.compile(
    rf"^(?:{_MISSING_TOKEN_SEPARATOR})*"
    rf"(?:{_MISSING_TOKEN_ALTERNATIVES})"
    rf"(?:(?:{_MISSING_TOKEN_SEPARATOR})(?:{_MISSING_TOKEN_ALTERNATIVES}))*"
    rf"(?:{_MISSING_TOKEN_SEPARATOR})*$",
    re.IGNORECASE,
)
_CORP_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "corporation",
    "corp",
    "inc",
    "limited",
    "ltd",
    "co",
)
_ANON_COMPACT_PUNCT = re.compile(
    r"[\s\u3000·•,，.。:：;；()（）\[\]【】{}<>《》'\"“”‘’_\-/]+"
)
_LABEL_TOKEN = r"(?:[A-J](?:[1-9])?|[1-9]|[一二三四五六七八九十甲乙丙丁戊己庚辛壬癸])"
_POSITION_LABEL = rf"(?:第)?{_LABEL_TOKEN}(?:名|大|位|号)?"
_ENTITY_LABEL = (
    r"(?:客户|客戶|供应商|供應商|公司|单位|單位|法人|自然人|个人|個人|经销商|經銷商)"
)
_ANON_PATTERNS = (
    re.compile(rf"^{_POSITION_LABEL}$", re.IGNORECASE),
    re.compile(
        rf"^(?:销售|銷售|采购|採購|经销商|經銷商|主要|保密|涉密|某涉密)?"
        rf"{_ENTITY_LABEL}{_POSITION_LABEL}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_POSITION_LABEL}(?:名|大|位|号)?{_ENTITY_LABEL}$",
        re.IGNORECASE,
    ),
    re.compile(rf"^(?:收入|销售|銷售|采购|採購){_POSITION_LABEL}$"),
    re.compile(
        r"^(?:某|主要|销售|銷售|采购|採購|经销商|經銷商|保密|涉密|某涉密)?"
        rf"{_ENTITY_LABEL}$"
    ),
    re.compile(r"^同一实际控制人控制的(?:客户|客戶|供应商|供應商)$"),
    re.compile(
        r"^(?:customer|supplier|company|counterparty|unit)"
        r"(?:[A-J](?:[1-9])?|[1-9])?$",
        re.IGNORECASE,
    ),
)
_SHORT_SOME_PLACEHOLDER = re.compile(
    r"^[\u3400-\u9fff]{0,4}某(?:客户|客戶|供应商|供應商|公司|单位|單位|"
    r"有限公司|有限责任公司|股份有限公司|集团有限公司)$"
)
_SPECIAL_TREATMENT_PREFIX = re.compile(r"^\s*(?:s\s*)?\*\s*st", re.IGNORECASE)
_NUMERIC_MULTIPLICATION = re.compile(r"(?<=\d)\s*[*×]\s*(?=\d)")
_MASK_SYMBOL = re.compile(r"[*×]")


class CounterpartyNameClass(str, Enum):
    """Semantic class used consistently at extraction and resolution boundaries."""

    NAMED = "named"
    ANONYMOUS = "anonymous"
    INVALID = "invalid"


def _normalized_text(value: object) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def is_redacted_counterparty(value: object) -> bool:
    """Return whether ``value`` is a pure dash/star redaction marker."""

    compact = re.sub(r"\s+", "", _normalized_text(value))
    return bool(
        compact and (_DASH_ONLY.fullmatch(compact) or _STAR_ONLY.fullmatch(compact))
    )


def _contains_structural_mask(value: str) -> bool:
    """Return whether a star/cross is a name mask rather than market notation.

    Chinese listed-company display names legitimately use ``*ST`` and
    ``S*ST`` prefixes.  Dimensions and model specifications also commonly use
    ``*`` or ``×`` between digits.  Neither form is evidence of anonymisation;
    any remaining mask symbol is treated conservatively as redaction.
    """

    without_market_prefix = _SPECIAL_TREATMENT_PREFIX.sub("", value, count=1)
    without_multiplication = _NUMERIC_MULTIPLICATION.sub("", without_market_prefix)
    return bool(_MASK_SYMBOL.search(without_multiplication))


def classify_counterparty_name(value: object) -> CounterpartyNameClass:
    """Classify a raw counterparty label without attempting entity resolution.

    Redaction markers are semantically anonymous.  The extraction layer adds
    the separate requirement that a pure marker carry a positive disclosed
    amount or share, which distinguishes real masked disclosures from blank
    table rows.
    """

    normalized = _normalized_text(value)
    if not normalized:
        return CounterpartyNameClass.INVALID

    if _MISSING_NAME.fullmatch(normalized):
        return CounterpartyNameClass.INVALID
    if is_redacted_counterparty(normalized):
        return CounterpartyNameClass.ANONYMOUS
    if _contains_structural_mask(normalized):
        return CounterpartyNameClass.ANONYMOUS

    compact = _ANON_COMPACT_PUNCT.sub("", normalized)
    if any(pattern.fullmatch(compact) for pattern in _ANON_PATTERNS):
        return CounterpartyNameClass.ANONYMOUS
    if _SHORT_SOME_PLACEHOLDER.fullmatch(compact):
        return CounterpartyNameClass.ANONYMOUS
    if any(character.isalnum() for character in normalized):
        return CounterpartyNameClass.NAMED
    return CounterpartyNameClass.INVALID


def normalize_company_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = _PUNCT.sub("", text)
    for suffix in sorted(_CORP_SUFFIXES, key=len, reverse=True):
        normalized_suffix = _PUNCT.sub("", suffix.lower())
        if text.endswith(normalized_suffix) and len(text) > len(normalized_suffix) + 1:
            text = text[: -len(normalized_suffix)]
            break
    return text


def is_anonymous_counterparty(value: object) -> bool:
    return classify_counterparty_name(value) is CounterpartyNameClass.ANONYMOUS
