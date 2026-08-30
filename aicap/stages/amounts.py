#!/usr/bin/env python3
"""Stage `normalize_amounts` -- deterministic-first magnitude normalization.

The rule layer resolves qualifier semantics (up to / at least / about / over
/ > / <) and per-year / per-unit period scopes -- but ONLY when the cue binds
syntactically inside amount_text itself.  Everything the rules cannot bind
goes to the LLM under the same contract (native currency units, every digit
reconstructible from the source text).

Outputs one row per observation: obs_amounts.jsonl.

NOTE on amount_method: a row that was ESCALATED to the LLM is stamped
amount_method="llm" even when the LLM abstained and the deterministic values
were kept (the amount_flags then carry the *_det_fallback reason).  Read
"llm" as "the LLM saw this row", not "the LLM produced these numbers".
`amount_value_source` disambiguates: "llm" iff the LLM's numbers were
accepted, "rule_fallback" when an escalated row kept the deterministic
values, "rule" for un-escalated rows, "none" when nothing parsed.  The
quality gate BLOCKS (review_required) every fallback-failure outcome:
escalation means the rules distrusted their own values, so keeping them
after an LLM failure is never a clean canonical amount.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .common import (Manifest, coerce_num, is_nullish, read_jsonl,
                     write_jsonl)
from .judge import Judge, content_id

SCALE = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "million": 1e6, "mm": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
SCALE.update({"crore": 1e7, "lakh": 1e5, "tn": 1e12})
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CCY_NUM_RE = re.compile(r"[$€£¥]\s*(\d[\d,]*(?:\.\d+)?)")
RANGE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:million|billion|bn|mm|mn|m|trillion|crore|lakh|tn|k|thousand)?"
                      r"\s*(?:to|through|[-–—])\s*[$€£¥]?\s*(\d[\d,]*(?:\.\d+)?)")
# 'and' as a range connector ONLY under an explicit 'between' ("$5bn debt and
# $4bn equity" must never read as a range)
BETWEEN_RE = re.compile(r"between\s+[$€£¥]?\s*(\d[\d,]*(?:\.\d+)?)\s*"
                        r"(?:million|billion|bn|mm|mn|m|trillion|crore|lakh|tn|k|thousand)?\s*"
                        r"and\s+[$€£¥]?\s*(\d[\d,]*(?:\.\d+)?)", re.I)
SCALEWORD_RE = re.compile(r"\b(trillion|billion|bn|b|million|mm|mn|m|thousand|crore|lakh|tn|k)\b")

# AMT-1: a parenthetical USD conversion beside a native-currency claim
# ("450 trillion won ($310.79 billion)").  The rule layer reads the FIRST
# scale word and the $-prefixed numeral, welding the native scale onto the
# converted number -- the $312T phantom.  The LLM lane demonstrably handles
# the construct, so these rows escalate.
PAREN_CONV_RE = re.compile(r"\(\s*(?:US)?\$\s*[0-9]", re.I)
TWO_CCY_RE = re.compile(r"(won|yen|yuan|rmb|rupee|dirham|riyal|rand|euro|₩|¥|€|£)"
                        r".{0,40}\(\s*(?:US)?\$", re.I)
# AMT-2: the parenthetical's own number + scale ("($310.79 billion)",
# "(US$46m)").  Deterministic capture of the SOURCE's conversion -- never an
# FX-table invention, consistent with AMOUNT_SYS's do-NOT-convert policy.
PAREN_USD_RE = re.compile(r"\(\s*(?:US)?\$\s*([0-9][0-9,.]*)\s*"
                          r"(trillion|billion|million|thousand|bn|mm|mn|[tbmk])\b", re.I)

# AMT-4: broad deterministic currency recovery. The extractor's historical
# enum collapsed KRW/AUD/CAD/SGD/IDR/HKD/... to `other`; normalize the adjacent
# textual unit before fact dedup so unlike currencies never share an amount
# identity group. A currency word without an adjacent matching magnitude does
# not count ("won" as a verb is a known false positive).
_CUR_ALIASES = {
    "won": "KRW", "krw": "KRW", "₩": "KRW",
    "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY", "cny": "CNY",
    "yen": "JPY", "jpy": "JPY", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "gbp": "GBP",
    "rupee": "INR", "rupees": "INR", "inr": "INR", "₹": "INR",
    "a$": "AUD", "aud": "AUD", "c$": "CAD", "cad": "CAD",
    "s$": "SGD", "sgd": "SGD", "hk$": "HKD", "hkd": "HKD",
    "nz$": "NZD", "nzd": "NZD", "idr": "IDR", "rupiah": "IDR",
    "chf": "CHF", "brl": "BRL", "zar": "ZAR", "thb": "THB",
    "myr": "MYR", "php": "PHP", "mxn": "MXN", "ils": "ILS",
    "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "us$": "USD", "$": "USD", "usd": "USD", "€": "EUR", "£": "GBP",
}
_CUR_TOKEN = "(?:" + "|".join(sorted((re.escape(k) for k in _CUR_ALIASES),
                                      key=len, reverse=True)) + ")"
_CUR_SUFFIX_RE = re.compile(
    rf"(?P<num>{NUM_RE.pattern})\s*(?P<scale>trillion|billion|million|thousand|"
    rf"crore|lakh|tn|bn|mn|mm|[tbmk])?\s*(?P<cur>{_CUR_TOKEN})(?![A-Za-z])", re.I)
_CUR_PREFIX_RE = re.compile(
    rf"(?P<cur>{_CUR_TOKEN})\s*(?P<num>{NUM_RE.pattern})\s*"
    rf"(?P<scale>trillion|billion|million|thousand|crore|lakh|tn|bn|mn|mm|[tbmk])?\b", re.I)
_CUR_SCALE = dict(SCALE, crore=1e7, lakh=1e5, tn=1e12)

# qualifier -> bound type, matched ONLY as a prefix directly before the money
# expression inside amount_text (syntactic binding).
_Q_MAX = r"(?:up[\s-]+to|as\s+much\s+as|no\s+more\s+than|maximum\s+of|<=?)"
_Q_MIN = r"(?:at\s+least|more\s+than|over|above|exceed(?:s|ing)?|upwards\s+of|>=?)"
_Q_APPROX = r"(?:approximately|approx\.?|about|around|roughly|some|nearly|almost|an?\s+estimated|~)"
_PREFIX_RE = re.compile(
    rf"^\s*(?:(?P<max>{_Q_MAX})|(?P<min>{_Q_MIN})|(?P<approx>{_Q_APPROX}))\s*[$€£¥]?\s*\d",
    re.I)
_ANY_QUALIFIER_RE = re.compile(
    rf"\b(?:{_Q_MAX}|{_Q_MIN}|{_Q_APPROX}|up to|at least|between|per)\b|~|[<>]", re.I)
_PER_YEAR_RE = re.compile(r"\b(?:annual(?:ly|ised|ized)?|per\s+year|per\s+annum|"
                          r"each\s+year|a\s+year|/(?:yr|year))\b", re.I)
_PER_UNIT_RE = re.compile(r"\bper\s+(?:\w+\s+)?(?:mw|gw|megawatt|gigawatt|unit|share|"
                          r"chip|gpu|server|rack|sq(?:uare)?\s*f(?:oo|ee)?t)\b", re.I)

AMOUNT_SYS = """You are the regv1 AMOUNT NORMALIZER. Given an amount and its context, emit the numeric value(s) it denotes, in FULL NATIVE CURRENCY UNITS (do NOT convert currencies). Use ONLY digits that appear in the amount text or the quote, scaled by the stated scale word (thousand/million/billion/trillion/crore/lakh). NEVER invent a number that is not reconstructible from the source digits.

Rules: a single figure -> amount_point (leave low/high null). A genuine RANGE ('$40-60bn') -> amount_low and amount_high, amount_point NULL. 'up to X' -> bound_type=max with amount_high=X. 'at least / more than X' -> bound_type=min with amount_low=X. '~ / approximately X' -> bound_type=approx, amount_point=X. A per-unit rate ('$X per MW') -> period_scope=per_unit. A per-year figure -> period_scope=per_year. A total -> period_scope=total. If the amount text does not denote a monetary amount at all, return all-null with bound_type=unclear.

Reply JSON ONLY: {"amount_point": <num|null>, "amount_low": <num|null>, "amount_high": <num|null>, "bound_type": "exact|approx|min|max|range|unclear", "period_scope": "total|per_year|per_unit|unclear"}"""

_MULTS = (1.0, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12)


def _to_float(x) -> float:
    return float(str(x).rstrip(",").replace(",", ""))


def _is_year(n: float) -> bool:
    return 1900 <= n <= 2099 and n == int(n)


def reconstruct_ok(nums, source) -> bool:
    """Every emitted number must be a source digit-string scaled by 10^{0,3,6,9,12}."""
    bases = [_to_float(x) for x in NUM_RE.findall(str(source))]
    if not bases:
        return not nums
    for x in nums:
        ax = abs(float(x))
        tol = max(1.0, ax * 1e-4)
        if not any(abs(ax - abs(b) * m) <= tol for b in bases for m in _MULTS):
            return False
    return True


def _valid_amount(o) -> bool:
    if not isinstance(o, dict):
        return False
    for k in ("amount_point", "amount_low", "amount_high"):
        if o.get(k) is not None and not isinstance(o.get(k), (int, float)):
            return False
    return (o.get("bound_type") in {"exact", "approx", "min", "max", "range", "unclear"}
            and o.get("period_scope") in {"total", "per_year", "per_unit", "unclear"})


def _det_rep(det: dict):
    p, lo, hi = (det.get("amount_native_point"), det.get("amount_native_low"),
                 det.get("amount_native_high"))
    if p is not None:
        return float(p)
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    return float(hi if hi is not None else lo) if hi is not None or lo is not None else None


def _currency_mentions(text: str):
    out = []
    for rx in (_CUR_PREFIX_RE, _CUR_SUFFIX_RE):
        for match in rx.finditer(str(text or "")):
            value = _to_float(match.group("num"))
            value *= _CUR_SCALE.get((match.group("scale") or "").lower(), 1.0)
            currency = _CUR_ALIASES.get(match.group("cur").lower())
            if currency:
                out.append((currency, value, match.group(0)))
    return out


def resolve_amount_currency(row: dict, det: dict):
    """Return (ISO currency, method, evidence, raw/conflict flag)."""
    raw = str(row.get("currency") or "unknown").strip().upper() or "UNKNOWN"
    native = _det_rep(det)
    text = f"{row.get('amount_text') or ''} {row.get('verbatim_quote') or ''}"
    matches = []
    if native is not None and native > 0:
        matches = [(cur, evidence) for cur, value, evidence in _currency_mentions(text)
                   if value > 0 and abs(value - native) / native <= 0.02]
    currencies = {cur for cur, _ in matches}
    spoken = next(iter(currencies)) if len(currencies) == 1 else None
    evidence = next((ev for cur, ev in matches if cur == spoken), None)
    valid_raw = raw not in {"OTHER", "UNKNOWN", ""}
    if spoken and (not valid_raw or spoken != raw):
        return spoken, ("quote_recovered" if not valid_raw else "quote_override"), \
            evidence, bool(valid_raw and spoken != raw)
    return raw, "extractor_tag", evidence, False


def recover_quote_usd(row: dict, det: dict):
    """Recover an adjacent source USD conversion omitted from amount_text."""
    if det.get("amount_usd_reported") is not None:
        return det["amount_usd_reported"], det.get("usd_reported_source")
    quote = str(row.get("verbatim_quote") or "")
    amount_text = str(row.get("amount_text") or "")
    native_num = NUM_RE.search(amount_text)
    if not quote or not native_num:
        return None, None
    needle = native_num.group(0).replace(",", "")
    candidates = []
    for match in PAREN_USD_RE.finditer(quote):
        prefix = quote[max(0, match.start() - 120):match.start()]
        if needle not in prefix.replace(",", ""):
            continue
        scale = SCALE.get(match.group(2).lower())
        if scale:
            candidates.append(_to_float(match.group(1)) * scale)
    values = {float(value) for value in candidates}
    if len(values) == 1:
        return values.pop(), "source_conversion_quote_recovered"
    return None, None


def parse_amount_rule(amount_text, amount_scale) -> dict:
    """Deterministic layer. Returns normalized fields + `resolved` flag +
    `escalate_reasons` (non-empty -> the row goes to the LLM)."""
    out = {"amount_native_point": None, "amount_native_low": None,
           "amount_native_high": None, "amount_bound_type": "unclear",
           "amount_period_scope": "unclear", "resolved": False,
           "amount_usd_reported": None, "usd_reported_source": None,
           "escalate_reasons": [], "flags": []}
    if is_nullish(amount_text):
        out["resolved"] = True                 # capacity-only row: nothing to parse
        return out
    text = str(amount_text)
    low = text.lower()
    sc = (amount_scale or "").strip().lower()

    # AMT-2: capture the source's own USD conversion from the parenthetical,
    # on EVERY lane (the field is deterministic from amount_text; the LLM is
    # never asked for it).  Requires an explicit scale token -- a bare
    # "($1,200)" is too ambiguous to scale.
    pm = PAREN_USD_RE.search(text)
    if pm:
        pscale = SCALE.get(pm.group(2).lower())
        if pscale:
            out["amount_usd_reported"] = _to_float(pm.group(1)) * pscale
            out["usd_reported_source"] = "source_conversion"

    sw = SCALEWORD_RE.search(low)
    if sw:
        mult = SCALE.get(sw.group(1), 1.0)
        if sc == "full_units":
            out["flags"].append("scale_conflict_text_wins")
    elif sc == "full_units":
        mult = 1.0
    elif sc in SCALE:
        mult = SCALE[sc]
    else:
        mult = 1.0
    scale_claimed = bool(sw) or sc in SCALE

    def to_native(v: float) -> float:
        if v >= 1e6 and scale_claimed and mult > 1:
            return v                           # already full-dollar in the text
        return v * mult

    ccy = [_to_float(x) for x in CCY_NUM_RE.findall(text)]
    alln = [_to_float(x) for x in NUM_RE.findall(re.sub(r"[$€£¥]", " ", text))]
    cand = ccy if ccy else ([n for n in alln if not _is_year(n)] or alln)
    if not cand:
        out["escalate_reasons"].append("parse_failed")
        return out

    # period scope from amount_text-local cues (never the whole quote)
    if _PER_UNIT_RE.search(low):
        out["amount_period_scope"] = "per_unit"
    elif _PER_YEAR_RE.search(low):
        out["amount_period_scope"] = "per_year"

    # AMT-1: explicit parenthetical-conversion escalation.  Two currencies (or
    # a native scale word) plus a "($..." conversion means the rule layer's
    # first-scale-word + $-numeral welding is exactly the wrong machine.
    if PAREN_CONV_RE.search(text) and (TWO_CCY_RE.search(text) or SCALEWORD_RE.search(low)):
        out["escalate_reasons"].append("parenthetical_conversion")
        out["resolved"] = False

    rng = BETWEEN_RE.search(text) or RANGE_RE.search(text)
    # AMT-1: count ALL non-year numbers, not just currency-prefixed ones --
    # the $-preferring count saw "450 trillion won ($310.79 billion)" as ONE
    # number and never fired the multiple_numbers guard.  (The Stage-2 gate
    # parser always counted all tokens; this brings the registry parser level.)
    # The max() keeps the OLD count as a floor: the year filter reads a
    # monetary "2,000" as the year 2000 ("A$1,800 MILLION TO A$2,000 MILLION"
    # would drop to one counted number and stop escalating).  Neither signal
    # may ever LOWER the guard.
    numbers_used = max(len(ccy),
                       len([n for n in alln if not _is_year(n)]) or len(alln))
    if rng:
        pair = sorted(to_native(_to_float(g)) for g in rng.groups())
        out.update({"amount_native_low": pair[0], "amount_native_high": pair[-1],
                    "amount_bound_type": "range", "resolved": True})
        if numbers_used > 2:
            out["escalate_reasons"].append("extra_numbers_beyond_range")
            out["resolved"] = False
        # AMT-3: both endpoints take the SINGLE first scale word, so
        # "$40 million to $60 billion" silently mis-scales one endpoint.
        # Two DIFFERENT scale magnitudes in the text -> the rule result is
        # untrustworthy -> LLM lane.
        mults_seen = {SCALE[w] for w in SCALEWORD_RE.findall(low) if w in SCALE}
        if len(mults_seen) > 1:
            out["escalate_reasons"].append("range_mixed_scale")
            out["resolved"] = False
        return out

    value = to_native(cand[0])
    m = _PREFIX_RE.match(text)
    if m:
        if m.group("max"):
            out.update({"amount_native_high": value, "amount_bound_type": "max"})
        elif m.group("min"):
            out.update({"amount_native_low": value, "amount_bound_type": "min"})
        else:
            out.update({"amount_native_point": value, "amount_bound_type": "approx"})
        out["resolved"] = True
    elif text.strip().startswith(">"):
        out.update({"amount_native_low": value, "amount_bound_type": "min",
                    "resolved": True})
    elif text.strip().startswith("<"):
        out.update({"amount_native_high": value, "amount_bound_type": "max",
                    "resolved": True})
    else:
        out.update({"amount_native_point": value, "amount_bound_type": "exact",
                    "resolved": True})

    # residual ambiguity -> escalate (precision first)
    tail_qual = _ANY_QUALIFIER_RE.search(text[m.end():] if m else text[1:])
    if out["resolved"] and m is None and _ANY_QUALIFIER_RE.search(text) \
            and out["amount_bound_type"] == "exact" \
            and not (_PER_YEAR_RE.search(low) or _PER_UNIT_RE.search(low)):
        # a qualifier somewhere in amount_text that did NOT bind as a prefix
        out["escalate_reasons"].append("unbound_qualifier_in_text")
        out["resolved"] = False
    elif out["resolved"] and m is not None and tail_qual:
        out["escalate_reasons"].append("second_qualifier")
        out["resolved"] = False
    if numbers_used > 1 and not rng:
        out["escalate_reasons"].append("multiple_numbers")
        out["resolved"] = False
    return out


def aggregation_escalation(level, det: dict) -> bool:
    """unit_rate without a detected per-cue is too ambiguous for rules."""
    return (str(level or "").lower() == "unit_rate"
            and det.get("amount_period_scope") != "per_unit")


_PREFIX_CLASS_RES = [("max", re.compile(rf"(?:{_Q_MAX})\s*[$€£¥]?\s*$", re.I)),
                     ("min", re.compile(rf"(?:{_Q_MIN})\s*[$€£¥]?\s*$", re.I)),
                     ("approx", re.compile(rf"(?:{_Q_APPROX})\s*[$€£¥]?\s*$", re.I))]
_SUFFIX_PER_RE = re.compile(r"^\s*(?:billion|million|bn|mm|mn|m|k|thousand|trillion)?\s*"
                            r"(?:dollars?\s+)?(?:a\s+year|per\s+(?:year|annum|mw|gw|"
                            r"megawatt|gigawatt|unit|share)|annually|each\s+year)", re.I)


def _quote_bound_qualifier(row: dict, det: dict) -> Optional[str]:
    """SYNTACTICALLY BOUND qualifier in the quote that amount_text did not
    carry. Returns an escalation reason or None. 'over the next four years'
    AFTER the number never triggers; a prefix qualifier that agrees with the
    bound the rules already produced never triggers either."""
    at, quote = row.get("amount_text"), row.get("verbatim_quote") or ""
    if is_nullish(at) or not quote:
        return None
    m = NUM_RE.search(str(at))
    if not m:
        return None
    idx = quote.find(m.group(0))
    if idx < 0:
        return None
    prefix = quote[max(0, idx - 40): idx]
    for cls, rx in _PREFIX_CLASS_RES:
        if rx.search(prefix):
            if det.get("amount_bound_type") != cls:
                return f"quote_prefix_qualifier:{cls}"
            return None
    tail = quote[idx + len(m.group(0)): idx + len(m.group(0)) + 40]
    if _SUFFIX_PER_RE.match(tail) and det.get("amount_period_scope") == "unclear":
        return "quote_suffix_per_cue"
    return None


def run(outdir: str, judge: Judge) -> dict:
    man = Manifest(outdir, "normalize_amounts")
    obs_path = os.path.join(outdir, "observations.jsonl")
    man.add_input(obs_path)
    rows = read_jsonl(obs_path)

    results: dict[str, dict] = {}
    escalations: list[tuple[str, dict, dict]] = []
    for row in rows:
        rid = row["raw_id"]
        det = parse_amount_rule(row.get("amount_text"), row.get("amount_scale"))
        reasons = list(det["escalate_reasons"])
        if det["resolved"]:
            qb = _quote_bound_qualifier(row, det)
            if qb:
                reasons.append(qb)
            if aggregation_escalation(row.get("amount_aggregation_level"), det):
                reasons.append("unit_rate_level_no_cue")
        # amount + capacity on one row: only ambiguous when a per-unit cue is
        # in reach; otherwise the parsed total stands (digits still guarded).
        if det["resolved"] and not is_nullish(row.get("amount_text")) \
                and not is_nullish(row.get("quantity_value")) \
                and re.search(r"\bper\b", str(row.get("amount_text") or ""), re.I):
            reasons.append("amount_and_quantity_per_cue")
        # digit reconstruction guard on the rule output
        if det["resolved"]:
            nums = [v for v in (det["amount_native_point"], det["amount_native_low"],
                                det["amount_native_high"]) if v is not None]
            src = f"{row.get('amount_text') or ''} {row.get('verbatim_quote') or ''}"
            if nums and not reconstruct_ok(nums, src):
                reasons.append("rule_not_reconstructible")
        if reasons and not is_nullish(row.get("amount_text")):
            escalations.append((rid, row, det))
            det = dict(det)
            det["escalate_reasons"] = reasons
        results[rid] = det

    # ---- LLM gray zone (batched) ----------------------------------------
    requests, req_row = [], {}
    for rid, row, det in escalations:
        user = "\n".join([
            f"amount_text: {row.get('amount_text')}",
            f"amount_scale: {row.get('amount_scale')}",
            f"currency: {row.get('currency')}",
            f"amount_basis: {row.get('amount_basis')}",
            f"amount_aggregation_level: {row.get('amount_aggregation_level')}",
            f"quantity: {row.get('quantity_value')} {row.get('quantity_unit') or ''}",
            f"method_summary: {row.get('method_summary')}",
            f"verbatim_quote: {row.get('verbatim_quote')}"])
        item = content_id("amt", rid, user)
        requests.append((item, user))
        req_row[item] = (rid, row)
    answers = judge.batch_json("r1_amount", requests, AMOUNT_SYS,
                               validate=_valid_amount, label="r1-amount")

    llm_ok = llm_abstain = 0
    for item, obj in answers.items():
        rid, row = req_row[item]
        det = results[rid]
        det["amount_method"] = "llm"
        # amount_value_source records WHO produced the numbers actually kept:
        # "llm" only when the LLM's answer was accepted; a fallback to the
        # deterministic values is "rule_fallback" (amount_method="llm" alone
        # only ever meant "the LLM saw this row").
        if obj is None:
            det["flags"].append("llm_quarantine_det_fallback")
            det["amount_value_source"] = "rule_fallback"
            llm_abstain += 1
            continue
        nums = [obj.get(k) for k in ("amount_point", "amount_low", "amount_high")]
        nums = [n for n in nums if isinstance(n, (int, float))]
        src = f"{row.get('amount_text') or ''} {row.get('verbatim_quote') or ''}"
        if nums and not reconstruct_ok(nums, src):
            det["flags"].append("llm_not_reconstructible_det_fallback")
            det["amount_value_source"] = "rule_fallback"
            llm_abstain += 1
            continue
        if nums:
            det.update({"amount_native_point": obj.get("amount_point"),
                        "amount_native_low": obj.get("amount_low"),
                        "amount_native_high": obj.get("amount_high"),
                        "amount_bound_type": obj.get("bound_type") or "unclear",
                        "amount_period_scope": obj.get("period_scope") or "unclear"})
            det["amount_value_source"] = "llm"
            llm_ok += 1
        else:
            det["flags"].append("llm_all_null")
            det["amount_value_source"] = "rule_fallback"
            llm_abstain += 1

    out_rows = []
    n_rule = n_llm = n_failed = 0
    for row in rows:
        rid = row["raw_id"]
        det = results[rid]
        method = det.get("amount_method") or ("rule" if det["resolved"] else "failed")
        has_amount = not is_nullish(row.get("amount_text"))
        if method == "rule" and has_amount:
            n_rule += 1
        elif method == "llm":
            n_llm += 1
        elif has_amount:
            n_failed += 1
        currency, currency_method, currency_evidence, currency_conflict = \
            resolve_amount_currency(row, det)
        usd_reported, usd_source = recover_quote_usd(row, det)
        # AMT-3: materialize the midpoint from the FINAL values (rule or LLM):
        # true midpoint when both endpoints exist, else the point.  max/min
        # bounds carry no midpoint -- the stated value is a bound, not a
        # center.  representative_amount (fact identity) is deliberately
        # untouched.
        p, lo, hi = (det["amount_native_point"], det["amount_native_low"],
                     det["amount_native_high"])
        mid = (lo + hi) / 2.0 if (lo is not None and hi is not None) else p
        out_rows.append({
            "raw_id": rid,
            "amount_native_point": det["amount_native_point"],
            "amount_native_low": det["amount_native_low"],
            "amount_native_high": det["amount_native_high"],
            "amount_native_mid": mid,
            "amount_usd_reported": usd_reported,
            "usd_reported_source": usd_source,
            "amount_currency": currency,
            "amount_currency_raw": (row.get("currency") or "unknown"),
            "amount_currency_method": currency_method,
            "amount_currency_evidence": currency_evidence,
            "amount_currency_conflict": currency_conflict,
            "amount_bound_type": det["amount_bound_type"],
            "amount_period_scope": det["amount_period_scope"],
            "amount_method": method if has_amount else "no_amount",
            "amount_value_source": (
                (det.get("amount_value_source") or "rule") if method == "llm"
                else "rule" if method == "rule" and has_amount
                else "none"),
            "amount_flags": sorted(set(det["flags"] + (det["escalate_reasons"] or []))),
        })
    path = os.path.join(outdir, "obs_amounts.jsonl")
    n = write_jsonl(out_rows, path)
    man.add_output(path, n)
    stats = {"rows": n, "rule_resolved": n_rule, "llm_resolved": n_llm,
             "llm_requests": len(requests), "llm_ok": llm_ok,
             "llm_abstain_or_fallback": llm_abstain, "failed": n_failed,
             "no_amount_rows": sum(1 for r in out_rows if r["amount_method"] == "no_amount"),
             "usd_reported_rows": sum(1 for r in out_rows
                                      if r["amount_usd_reported"] is not None),
             "parenthetical_escalations": sum(1 for r in out_rows
                                              if "parenthetical_conversion" in r["amount_flags"]),
             "range_mixed_scale_escalations": sum(1 for r in out_rows
                                                  if "range_mixed_scale" in r["amount_flags"])}
    man.done(stats, judge.report())
    return stats
