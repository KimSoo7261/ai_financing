#!/usr/bin/env python3
"""Deterministic extraction-side helpers: amount parsing and target resolution.

The deterministic layer is MINIMAL.  All semantic judgments -- is-this-
revenue, AI-infra scope, instrument_class -- are made by the LLM (Stage 2
extracts + classifies; Stage 3 audits).  What remains here is only mechanical
work an LLM should not be trusted to do:

  A) AMOUNT PARSING  amount_text -> approximate USD billions, used INTERNALLY
     to gate "has a dollar?" and to raise amount_needs_review (never written
     to output, never summed).  A full-dollar number wrongly tagged with a
     scale word is NOT inflated; mass/tonnage and sq ft / acres are detected
     as non-currency; a power MW/GW figure is kept as a capacity.
  B) TARGET RESOLUTION  a filing/transcript is the filer's own book, so a
     null target DEFAULTS to the filer; news is attributed to the company the
     sentence names, so a null target in news stays null.
"""
from __future__ import annotations
import re
from typing import Optional

from .config import is_news_source

# ===========================================================================
# PART A -- AMOUNT PARSING  (internal; never summed across rows)
# ===========================================================================
_SCALE = {
    "trillion": 1e3, "t": 1e3,
    "billion": 1.0, "bn": 1.0, "b": 1.0,
    "million": 1e-3, "mm": 1e-3, "mn": 1e-3, "m": 1e-3,
    "thousand": 1e-6, "k": 1e-6,
    "crore": 1e-2, "lakh": 1e-4,
    "dollars": 1e-9, "dollar": 1e-9, "ones": 1e-9, "unit": 1e-9, "units": 1e-9,
}
# Scale WORDS that MULTIPLY a count up to billions. If the written number is
# ALREADY a full-dollar magnitude (>= 1e6) and one of these is claimed (e.g.
# "$10,000,000,000" tagged scale="billion"), the scale word is a model mislabel
# that would inflate ~1e9x. "dollars"/"ones"/"unit" are EXCLUDED (they divide).
_SCALE_UP = {"trillion", "t", "billion", "bn", "b", "million", "mm", "mn", "m",
             "thousand", "k", "crore", "lakh"}

_SQFT_RE  = re.compile(r"square\s*f(?:ee|oo)?t|\bsq\.?\s*ft\b", re.I)
_POWER_RE = re.compile(r"\b(?:mega|giga|kilo|tera)?watts?\b|\b[MGKT]?W\b|\b[MGKT]Wh\b", re.I)
_ACRE_RE  = re.compile(r"\bacres?\b", re.I)
# mass/tonnage is NOT currency ("623,000 tons" must not surface as $623bn).
_TONS_RE  = re.compile(r"\b(?:metric\s+|short\s+|long\s+)?(?:tonnes?|tons?)\b", re.I)
_NUM_RE   = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(text: str) -> Optional[float]:
    m = _NUM_RE.search(text.replace("$", ""))
    return float(m.group(0).rstrip(",").replace(",", "")) if m else None


def _count_number_tokens(text: str) -> int:
    return len(_NUM_RE.findall(text.replace("$", "")))


def classify_amount(amount_text: Optional[str],
                    amount_scale: Optional[str]) -> dict:
    """Auditable dict: amount_bn (approx, INTERNAL) / quantity_value /
    quantity_unit / needs_review / review_reason / method. Currency vs
    non-currency only -- revenue/D&A judgments are the LLM's job."""
    out = {"amount_bn": None, "quantity_value": None, "quantity_unit": None,
           "needs_review": False, "review_reason": None, "method": "none"}

    def _flag(reason: str):
        out["needs_review"] = True
        out["review_reason"] = (out["review_reason"] + ";" + reason
                                if out["review_reason"] else reason)

    if not amount_text:
        return out

    # which-number ambiguity: a packed table row carries several numbers
    # (current+prior columns, maturity ladders) -- reviewer must confirm WHICH.
    if _count_number_tokens(amount_text) >= 2:
        _flag("multiple_numbers_in_amount_text")

    val = _first_number(amount_text)
    sw = re.search(r"\b(trillion|billion|million|thousand|crore|lakh|bn|mm|mn)\b",
                   amount_text, re.I)
    phys_mult = ({"trillion": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6,
                  "mn": 1e6, "thousand": 1e3, "crore": 1e7,
                  "lakh": 1e5}.get(sw.group(1).lower(), 1.0) if sw else 1.0)

    # ---- non-currency physical quantities --------------------------------
    if _SQFT_RE.search(amount_text):
        out["quantity_value"] = val * phys_mult if val is not None else None
        out["quantity_unit"] = "sq_ft"; out["method"] = "non_currency_sqft"
        return out
    if _ACRE_RE.search(amount_text):
        out["quantity_value"] = val * phys_mult if val is not None else None
        out["quantity_unit"] = "acres"; out["method"] = "non_currency_acres"
        return out
    if _TONS_RE.search(amount_text) and "$" not in amount_text:
        out["quantity_value"] = val * phys_mult if val is not None else None
        out["quantity_unit"] = "tons"; out["method"] = "non_currency_tons"
        return out
    if _POWER_RE.search(amount_text) and "$" not in amount_text:
        unit = "MW"
        m2 = re.search(r"\b([MGKT])W\b", amount_text, re.I)
        spelled = re.search(r"\b(mega|giga|kilo|tera)watt", amount_text, re.I)
        if m2:
            unit = m2.group(1).upper() + "W"
        elif spelled:
            unit = {"mega": "MW", "giga": "GW", "kilo": "kW", "tera": "TW"}[spelled.group(1).lower()]
        out["quantity_value"] = val * phys_mult if val is not None else None
        out["quantity_unit"] = unit; out["method"] = "capacity_power"
        return out                       # a power commitment is real, not non-currency

    if val is None:
        return out

    # ---- currency -> approximate USD billions (INTERNAL) -----------------
    sc = (amount_scale or "").lower().strip()

    # "full_units" -- the number is already written in full currency units
    # (e.g. 13,300,000 -> $0.0133bn), no scale word to multiply up.  A scale
    # word WRITTEN IN amount_text beats the full_units tag: "$5 billion"
    # mistagged full_units must not silently deflate to $5.
    if sc == "full_units":
        if sw is not None:
            sw_key = sw.group(1).lower()
            sw_mult = _SCALE.get(sw_key)
            if sw_mult is not None:
                if val >= 1e6 and sw_key in _SCALE_UP:
                    # "$10,000,000,000 billion"-style absurdity: the number is
                    # already full dollars; keep full_units, flag the conflict.
                    out["amount_bn"] = val / 1e9
                    out["method"] = f"full_units_scale_word_ignored(conflict:{sw_key})"
                    _flag("scale_word_conflicts_full_dollar_number")
                    return out
                out["amount_bn"] = val * sw_mult
                out["method"] = f"text_scale_wins:{sw_key}(tagged:full_units)"
                _flag("scale_conflict_text_wins_over_full_units")
                return out
        out["amount_bn"] = val / 1e9
        out["method"] = "full_units"
        return out

    # full-dollar number that ALSO carries a multiply-up scale word -> mislabel
    # ("$10,000,000,000" tagged "billion" -> $10bn, flagged, not $1e10).
    claimed = sc if sc in _SCALE else (sw.group(1).lower() if sw else None)
    if val >= 1e6 and claimed in _SCALE_UP:
        out["amount_bn"] = val / 1e9
        out["method"] = f"full_dollars_scale_word_ignored(conflict:{claimed})"
        _flag("scale_word_conflicts_full_dollar_number")
        return out

    if sc in _SCALE:
        # A WRITTEN scale word in amount_text is authoritative over a
        # mis-tagged amount_scale: "$1.4 trillion" tagged scale="billion" must
        # not silently become $1.4bn.  When the written word maps to a
        # DIFFERENT multiplier than the tag, prefer the written word and flag;
        # unchanged when they agree or no word is present.
        if sw is not None:
            sw_key = sw.group(1).lower()
            sw_mult = _SCALE.get(sw_key)
            if sw_mult is not None and sw_mult != _SCALE[sc]:
                out["amount_bn"] = val * sw_mult
                out["method"] = f"text_scale_wins:{sw_key}(tagged:{sc})"
                _flag("scale_conflict_text_wins")
                return out
        out["amount_bn"] = val * _SCALE[sc]
        out["method"] = f"explicit_scale:{sc}"
        return out
    if sw:
        out["amount_bn"] = val * _SCALE.get(sw.group(1).lower(), 1.0)
        out["method"] = f"text_scale:{sw.group(1).lower()}"
        return out

    # bare number: disambiguate by magnitude (best-effort; flagged for review)
    _flag("scale_guessed")
    if val >= 1e7:
        out["amount_bn"] = val / 1e9
        out["method"] = "guess_full_dollars(>=1e7)"
    elif "," in amount_text:
        out["amount_bn"] = val / 1e3
        out["method"] = "guess_table_millions(comma)"
    else:
        out["method"] = "ambiguous_left_none"
    return out


# ===========================================================================
# PART B -- OBLIGOR RESOLUTION
# ===========================================================================
_FIRM_ALIASES = {
    "microsoft": "Microsoft", "msft": "Microsoft", "azure": "Microsoft",
    "amazon": "Amazon", "aws": "Amazon", "amazon web services": "Amazon", "amzn": "Amazon",
    "alphabet": "Alphabet", "google": "Alphabet", "google cloud": "Alphabet",
    "googl": "Alphabet", "goog": "Alphabet", "gcp": "Alphabet",
    "meta": "Meta", "facebook": "Meta", "fb": "Meta", "meta platforms": "Meta",
    "oracle": "Oracle", "orcl": "Oracle", "oci": "Oracle",
}


def canonical_firm(name: Optional[str]) -> Optional[str]:
    """Map a free-text company name to a canonical hyperscaler, else None."""
    if not name:
        return None
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).strip()
    if s in _FIRM_ALIASES:
        return _FIRM_ALIASES[s]
    for alias, canon in _FIRM_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", s):
            return canon
    return None


def _is_news(obs: dict) -> bool:
    # Shared predicate (config.is_news_source): reuters_news IS news.  The
    # local variant once matched only dcd_news/"news", so a null-target
    # Reuters observation fell through to the filer default and inherited the
    # chunk's firm tag -- exclusively big-5 hyperscalers -- inflating exactly
    # the aggregates the corpus exists to measure.
    return is_news_source(obs.get("source_type"))


# Target resolution only fills the filer-default for a null target so the
# gate does not drop the filer's own lines; a named entity is used AS WRITTEN
# (the extraction prompt bans merging to a parent).
def resolve_target(obs: dict) -> dict:
    """{target_firm}. A filing/transcript is the filer's own book: a null target
    DEFAULTS to the filer. News is attributed to the entity the sentence names --
    a null target in news stays null (the gate then drops it)."""
    target_raw = obs.get("target_firm")
    if target_raw:
        return {"target_firm": target_raw}          # as written; no hyperscaler forcing
    if not _is_news(obs):
        chunk_firm = obs.get("firm")
        return {"target_firm": canonical_firm(chunk_firm) or chunk_firm}   # filer default
    return {"target_firm": None}
