#!/usr/bin/env python3
"""Raw Stage-2 extractions -> gate -> the capital-instance table.

Turns raw observations into output rows (one row per raw observation; no
dedup -- deduplication is the Fact registry's job).  Two responsibilities:

  1. CLASSIFY (deterministic, mechanical): parse the amount internally (only
     to know "has a dollar?" and to raise amount_needs_review), resolve the
     TARGET, and carry the Stage-2 classification through unchanged.

  2. GATE (inclusion rules): keep iff there is a TARGET and a MAGNITUDE that
     can be placed -- a DOLLAR, or a physical power capacity (MW / GW; kept
     even without a named project or counterparty).  Everything that fails
     goes to dropped.jsonl with a drop_reason.  Semantic inclusion (scope,
     revenue, duplicates) is the Stage-3 auditor's job, not gated here.

STATISTICAL DISCIPLINE: amount_text mixes stock / forward magnitudes across
filings and must NEVER be summed across rows -- the parsed billions value is
internal and is not written out.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from typing import Optional

from .classify import classify_amount, resolve_target
from .config import (AMOUNT_AGGREGATION_LEVEL_SET, AMOUNT_BASIS_SET,
                     AMOUNT_BOUND_TYPE_SET, CLAIM_SOURCE_SET, DEBT_INSTRUMENTS,
                     DEBT_INSTRUMENT_SUBTYPE_SET, OBJECT_TYPE_SET,
                     PARENT_OBJECT_RELATION_SET, SCHEMA_VERSION,
                     TARGET_PARENT_RELATION_SET, canonical_instrument_class,
                     is_news_source)
from .prompts import (EVIDENCE_KEYS, _ground_norm, _span_grounded,
                      observation_defects)

_POWER_UNITS = {"MW", "GW", "KW", "TW"}

# Hosting-capacity marketing figures ("a site capable of supporting up to
# $70bn in AI data center investment") are NOT capital commitments. When the
# cue appears in the amount's own evidence, the amount is nulled (the row may
# still gate in on its power-capacity axis) and the original text is kept in
# the review reason. Cues are precision-first; the Stage-3 auditor remains the
# semantic backstop for phrasings this list misses.
_HOSTABLE_RE = re.compile(
    r"\b(?:capable\s+of\s+(?:support|host)\w*|capacity\s+to\s+(?:support|host)\w*|"
    r"(?:could|can|able\s+to)\s+(?:support|host)\b|"
    r"(?:support|host)(?:ing|s)?\s+up\s+to)\b", re.I)


def _hostable_amount(obs: dict) -> bool:
    """True iff the amount's evidence text carries a hosting-capacity cue.
    Scans only the amount-adjacent fields (field_evidence.amount_or_capacity,
    summary) -- NOT the whole quote -- to keep false positives near zero."""
    fe = obs.get("field_evidence") or {}
    if isinstance(fe, str):
        try:
            fe = json.loads(fe)
        except (ValueError, TypeError):
            fe = {}
    txt = " ".join(str(x) for x in (fe.get("amount_or_capacity"), obs.get("summary")) if x)
    return bool(_HOSTABLE_RE.search(txt))


def _coerce_amount_basis(raw) -> tuple:
    """Enum guard: an off-vocabulary amount_basis is coerced to "unknown" and
    flagged, never passed through. Returns (basis, coercion_flag | None)."""
    ab = (str(raw).strip().lower() if raw is not None else "")
    if not ab:
        return None, None
    if ab in AMOUNT_BASIS_SET:
        return ab, None
    return "unknown", f"invalid_amount_basis_coerced:{ab}"


def _coerce_enum(raw, allowed: set, *, fallback=None, field: str) -> tuple:
    """Closed-vocabulary guard used for the v2 aggregation fields."""
    value = str(raw).strip().lower() if raw is not None else ""
    if not value:
        return fallback, None
    if value in allowed:
        return value, None
    return fallback, f"invalid_{field}_coerced:{value}"


def _prep_row(obs: dict, defects: list | None = None) -> tuple:
    """Build the pre-review output row for one raw observation. Returns
    (row, amount_bn_internal). amount_bn_internal is used ONLY by the gate and is
    never written to output.

    `defects` is the observation's deterministic contract-defect list
    (prompts.observation_defects), recorded verbatim on the row as
    `schema_defects` -- 2026-08-28 policy: defects are admitted, defaulted
    or cleared here, RECORDED for the paper's error rates, and routed
    review_required at the quality gate; they are never repaired by
    re-prompting.  Missing keys read as None throughout this function, so
    the row-building is already default-tolerant."""
    amt = classify_amount(obs.get("amount_text"), obs.get("amount_scale"))
    who = resolve_target(obs)

    # Hostable-amount screen. Null the laundered "capable of supporting up to
    # $X" figure BEFORE the gate: the row then stands or falls on its capacity
    # axis (a real 2 GW substation keeps its capacity; a cue row with no
    # capacity drops as no_dollar_no_capacity, which is correct).
    if amt["amount_bn"] is not None and _hostable_amount(obs):
        obs = dict(obs)
        _orig_amt = obs.get("amount_text")
        obs["amount_text"] = None
        obs["amount_scale"] = None
        obs["amount_basis"] = None
        amt["amount_bn"] = None
        amt["needs_review"] = True
        amt["review_reason"] = ((amt["review_reason"] + ";") if amt["review_reason"]
                                else "") + f"hostable_amount_screened:{_orig_amt}"

    # Currency guard. A parsed amount in a non-USD currency is admitted but
    # flagged -- no FX conversion happens anywhere in the pipeline, so its
    # magnitude must not be read as USD without a human look. ("unknown" = no
    # currency stated in text; overwhelmingly USD in this corpus, unflagged.)
    _cur = (obs.get("currency") or "unknown").strip().upper()
    if amt["amount_bn"] is not None and _cur not in ("USD", "UNKNOWN"):
        amt["needs_review"] = True
        amt["review_reason"] = ((amt["review_reason"] + ";") if amt["review_reason"]
                                else "") + f"non_usd_unconverted:{_cur}"

    # Capacity comes from amount_text parsing OR directly from the model.
    qv = amt["quantity_value"] if amt["quantity_value"] is not None else obs.get("quantity_value")
    qu = amt["quantity_unit"] if amt["quantity_value"] is not None else obs.get("quantity_unit")

    # other_parties is a plain list of names.
    _op = obs.get("other_parties")
    _op = _op if isinstance(_op, list) else []

    # target_parent may be a string, a list of names (multi-sponsor JV/SPV), or null.
    _parent = obs.get("target_parent")
    if isinstance(_parent, list):
        _parent_out = json.dumps(_parent, ensure_ascii=False) if _parent else None
        _parent_present = bool(_parent)
    elif _parent and str(_parent).strip().lower() not in ("null", "none", "na"):
        _parent_out = _parent
        _parent_present = True
    else:
        _parent_out = None
        _parent_present = False
    _prel = (obs.get("target_parent_relation") or "").strip().lower()
    _prel = "na" if not _parent_present else (_prel if _prel in TARGET_PARENT_RELATION_SET and _prel != "na" else "unclear")

    _cond = (obs.get("conditionality") or "unclear").strip().lower()
    # Preserve the extracted condition span for EVERY conditionality value:
    # nulling it when conditionality != "conditional" destroyed evidence for
    # the "unclear"+span rows before the observation store (unrecoverable).
    _ctext = obs.get("condition_text")
    _cs = (obs.get("commitment_status") or "").strip().lower()
    _cs_cue = obs.get("commitment_status_cue") if _cs and _cs != "unclear" else None

    _basis, _basis_flag = _coerce_amount_basis(obs.get("amount_basis"))
    _bound, _bound_flag = _coerce_enum(
        obs.get("amount_bound_type"), AMOUNT_BOUND_TYPE_SET,
        fallback="unclear", field="amount_bound_type")
    _agg, _agg_flag = _coerce_enum(
        obs.get("amount_aggregation_level"), AMOUNT_AGGREGATION_LEVEL_SET,
        fallback="unknown", field="amount_aggregation_level")
    _claim, _claim_flag = _coerce_enum(
        obs.get("claim_source"), CLAIM_SOURCE_SET,
        fallback="unknown", field="claim_source")
    _obj_type, _obj_flag = _coerce_enum(
        obs.get("object_type"), OBJECT_TYPE_SET,
        fallback=None, field="object_type")
    _pobj_type, _pobj_flag = _coerce_enum(
        obs.get("parent_object_type"), OBJECT_TYPE_SET,
        fallback=None, field="parent_object_type")
    _pobj_rel, _pobj_rel_flag = _coerce_enum(
        obs.get("parent_object_relation"), PARENT_OBJECT_RELATION_SET,
        fallback=None, field="parent_object_relation")
    _inst = canonical_instrument_class(obs.get("instrument_class"))
    _debt, _debt_flag = _coerce_enum(
        obs.get("debt_instrument_subtype"), DEBT_INSTRUMENT_SUBTYPE_SET,
        fallback=("unknown" if _inst in DEBT_INSTRUMENTS else None),
        field="debt_instrument_subtype")
    if _inst not in DEBT_INSTRUMENTS and _debt is not None:
        _debt_flag = "nondebt_subtype_cleared"
        _debt = None
    if not obs.get("object_name"):
        if any(obs.get(k) is not None for k in
               ("object_type", "object_evidence", "parent_object_name",
                "parent_object_type", "parent_object_relation",
                "parent_object_evidence")):
            _obj_flag = "object_metadata_without_object_cleared"
        _obj_type = _pobj_type = _pobj_rel = None
    elif not obs.get("parent_object_name"):
        if any(obs.get(k) is not None for k in
               ("parent_object_type", "parent_object_relation", "parent_object_evidence")):
            _pobj_flag = "parent_object_metadata_without_name_cleared"
        _pobj_type = _pobj_rel = None
    # v2.1 pilot hotfix: extraction may emit a NAMED object with a null type
    # (the validator accepts it rather than forcing a guessed type or
    # quarantining the chunk's sibling observations).  The registry binds
    # only TYPED objects (DEC-013), so an untyped object -- or a parent
    # missing its type/relation -- is cleared here with a review flag; the
    # raw payload keeps the surface for audit.
    _obj_name = obs.get("object_name")
    _obj_ev = obs.get("object_evidence")
    _pobj_name = obs.get("parent_object_name")
    _pobj_ev = obs.get("parent_object_evidence")
    _untyped_flags = []
    if _obj_name is not None and _obj_type is None:
        _untyped_flags.append("untyped_object_cleared")
        _obj_name = _obj_ev = None
        _pobj_name = _pobj_ev = _pobj_type = _pobj_rel = None
    elif _pobj_name is not None and (_pobj_type is None or _pobj_rel is None):
        _untyped_flags.append("untyped_parent_object_cleared")
        _pobj_name = _pobj_ev = None
        _pobj_type = _pobj_rel = None
    elif _pobj_name is not None and any(
            str(x).startswith("parent_type_incompatible")
            or x == "parent_equals_object"
            for x in (defects or [])):
        # An incompatible (or self-referential) containment claim: the parent
        # annotation is cleared rather than trusted -- the registry would
        # refuse to materialize the edge anyway, and the defect list keeps
        # the model's original claim on record.
        _untyped_flags.append("incompatible_parent_object_cleared")
        _pobj_name = _pobj_ev = None
        _pobj_type = _pobj_rel = None
    _schema_flags = [f for f in (_basis_flag, _bound_flag, _agg_flag, _claim_flag,
                                  _obj_flag, _pobj_flag,
                                  _pobj_rel_flag, _debt_flag) if f]
    _schema_flags += _untyped_flags

    # Evidence grounding moved OUT of the extraction validator (2026-08-28
    # pilot decision: it must never trigger the repair/rescue lane -- the
    # opus-4.8 A/B proved the misses are contract-driven, not model-driven).
    # Every evidence span must still be real content of the observation's own
    # quote: an ungrounded span is CLEARED here and the field name recorded
    # in `evidence_cleared` (info-tier at the gate, never status-changing --
    # evidence is provenance; the field VALUES are independently audited by
    # Stage 3, and the quote itself remains verbatim-gated).
    _quote_norm = _ground_norm(obs.get("verbatim_quote") or "")
    _evidence_cleared = []
    _fe = obs.get("field_evidence") or {}
    if isinstance(_fe, str):
        try:
            _fe = json.loads(_fe)
        except (ValueError, TypeError):
            _fe = {}
    if not isinstance(_fe, dict):
        _fe = {}
    # Unknown evidence keys (the model over-evidencing beyond the 16
    # standard keys) are silently dropped here -- the defect list already
    # records them as evidence_unknown_keys.
    _fe = {k: v for k, v in _fe.items() if k in EVIDENCE_KEYS}
    for _k, _v in list(_fe.items()):
        if _v is not None and not isinstance(_v, str):
            _fe[_k] = None
            continue
        if _v is not None and not _span_grounded(_v, _quote_norm):
            _fe[_k] = None
            _evidence_cleared.append(f"field_evidence.{_k}")

    def _clear_ungrounded(value, label):
        if value is not None and not _span_grounded(value, _quote_norm):
            _evidence_cleared.append(label)
            return None
        return value

    _obj_ev = _clear_ungrounded(_obj_ev, "object_evidence")
    _pobj_ev = _clear_ungrounded(_pobj_ev, "parent_object_evidence")
    _ds_cue = _clear_ungrounded(obs.get("deal_status_cue"), "deal_status_cue")
    _cs_cue = _clear_ungrounded(_cs_cue, "commitment_status_cue")
    _ctext = _clear_ungrounded(_ctext, "condition_text")
    if _schema_flags:
        amt["needs_review"] = True
        joined = ";".join(_schema_flags)
        amt["review_reason"] = (amt["review_reason"] + ";" + joined
                                if amt["review_reason"] else joined)

    row = {
        "schema_version": SCHEMA_VERSION,
        "raw_id": obs.get("raw_id"),
        "target_firm": who["target_firm"],
        # A ';'-joined multi-entity target marks a COLLECTIVE observation (one
        # inseparable amount attributed to several named firms). Downstream
        # layers must never sum a collective amount into any single member.
        "target_is_multi_entity": bool(who["target_firm"] and ";" in str(who["target_firm"])),
        "target_parent": _parent_out,                 # string OR json list OR null
        "target_parent_relation": _prel,              # na iff target_parent null
        "amount_text": obs.get("amount_text"),
        "amount_scale": obs.get("amount_scale"),
        "currency": obs.get("currency"),
        "amount_basis": _basis,
        # The amount normalizer independently derives the authoritative final
        # amount_bound_type.  Preserve the extractor's answer for audit and
        # agreement telemetry instead of silently overwriting it.
        "amount_bound_type_extracted": _bound,
        "amount_aggregation_level": _agg,
        "quantity_value": qv,
        "quantity_unit": qu,
        "temporal_context": obs.get("temporal_context"),   # fact-local time evidence
        # ---- Stage-2 classification, carried through unchanged ----
        "instrument_class": _inst,
        "debt_instrument_subtype": _debt,
        "method_summary": obs.get("method_summary"),
        "object_name": _obj_name,
        "object_type": _obj_type,
        "object_evidence": _obj_ev,
        "parent_object_name": _pobj_name,
        "parent_object_type": _pobj_type,
        "parent_object_relation": _pobj_rel,
        "parent_object_evidence": _pobj_ev,
        "counterparty": obs.get("counterparty"),
        
        # Explicit economic-role resolution emitted by Stage 2.
        # Carried through unchanged for downstream direction checks.
        "economic_action": obs.get("economic_action"),
        "capital_provider": obs.get("capital_provider"),
        "capital_recipient": obs.get("capital_recipient"),
        
        "other_parties": json.dumps(_op, ensure_ascii=False),          # [name, ...] JSON
        "ai_infra_scope": obs.get("ai_infra_scope"),
        "geo_scope": obs.get("geo_scope"),
        "claim_source": _claim,
        "claim_source_name": obs.get("claim_source_name"),
        "deal_status": obs.get("deal_status"),
        "deal_status_cue": _ds_cue,
        "commitment_status": obs.get("commitment_status"),
        "commitment_status_cue": _cs_cue,             # null iff commitment_status unclear
        "conditionality": _cond,
        "condition_text": _ctext,                     # verbatim condition span (any conditionality)
        "summary": obs.get("summary"),
        "verbatim_quote": obs.get("verbatim_quote"),
        "field_evidence": json.dumps(_fe, ensure_ascii=False),
        # ungrounded evidence spans cleared at assembly (";"-joined field
        # names; empty when every span was grounded) -- info-tier signal
        "evidence_cleared": ";".join(_evidence_cleared),
        # deterministic contract defects (prompts.observation_defects) --
        # recorded for the paper's error rates; blocking at the gate
        "schema_defects": ";".join(defects or []),
        "amount_needs_review": amt["needs_review"],
        "amount_review_reason": amt["review_reason"],
        # Admission flags -- set at Stage-3 admission, defaulted here so the
        # columns exist on every row: audit_time_bad marks a time_ok-only bad
        # (admitted + flagged, not dropped); direction_fix_* record a direction
        # correction, where direction_fix_source separates a signed HUMAN
        # override (human_overrides.jsonl, reviewer set) from a deterministic
        # model-consensus swap (auto_direction_fixes.jsonl, reviewer null).
        "audit_time_bad": False,
        "direction_fix_applied": False,
        "direction_fix_source": None,
        "direction_fix_reviewer": None,
        # Explicit should=<different value> proposals from the audit note
        # (set at admission; admitted + review reason, never a drop).
        "audit_should_mismatch": None,
        # Stage 3 adds an independent audit table, linked by raw_id.
        "source_type": obs.get("source_type"),
        "doc_id": obs.get("doc_id"),
        "source_path": obs.get("source_path"),
        "doc_date": obs.get("doc_date"),
        "firm": obs.get("firm"),                       # the filer (fiscal-calendar fallback)
        "fiscal_period": obs.get("fiscal_period"),
        "chunk_id": obs.get("chunk_id"),
    }
    return row, amt["amount_bn"]


def _gate(row: dict, amount_bn: Optional[float]) -> tuple:
    """(keep, drop_reason). Keep iff there is a TARGET and:
       - a DOLLAR (amount_bn parsed), OR
       - a power-capacity magnitude (kept even without a named project or
         counterparty -- a GW commitment IS a real observation).
       Other non-currency magnitudes (sq ft / acres / tons) with no dollar,
       or no magnitude at all, are dropped."""
    if not row.get("target_firm"):
        return False, "no_target"
    if amount_bn is not None:
        return True, None
    qv = row.get("quantity_value")
    unit = (row.get("quantity_unit") or "").strip()
    if unit.upper() in _POWER_UNITS and qv is not None:      # power-capacity row -> keep
        return True, None
    if unit:                                                  # sq ft / acres / tons, no $
        return False, "non_currency_no_dollar"
    return False, "no_dollar_no_capacity"


def classify_and_gate(raw: list, *, include_news: bool = True) -> dict:
    """Return {'kept': [...], 'dropped': [...rows with drop_reason...],
    'defect_counts': {defect_class: n}, 'n_defective': int}.

    Contract defects are scanned deterministically per observation
    (prompts.observation_defects), recorded on every row, and tallied for
    the run report -- the paper's structural-error rates come from here.
    The one defect that still excludes an observation outright is a missing
    or empty verbatim_quote: with no quote there is nothing to audit, so the
    row goes to dropped.jsonl (reason no_verbatim_quote) instead of flowing
    unauditable."""
    kept, dropped = [], []
    defect_counts: Counter = Counter()
    n_defective = 0
    for o in raw:
        if not include_news and is_news_source(o.get("source_type")):
            continue
        defects = observation_defects(o)
        if defects:
            n_defective += 1
            for d in defects:
                defect_counts[d.split(":")[0]] += 1
        row, amount_bn = _prep_row(o, defects)
        no_quote = ("empty_verbatim_quote" in defects
                    or not (o.get("verbatim_quote") or "").strip())
        if no_quote:
            row = dict(row); row["drop_reason"] = "no_verbatim_quote"
            dropped.append(row)
            continue
        keep, reason = _gate(row, amount_bn)
        if keep:
            kept.append(row)
        else:
            row = dict(row); row["drop_reason"] = reason
            dropped.append(row)
    return {"kept": kept, "dropped": dropped,
            "defect_counts": dict(defect_counts), "n_defective": n_defective}


# ---------------------------------------------------------------------------
# Sampled drop spot-check lane. A dropped row that LOOKS like it should have
# survived -- a named target, a placeable magnitude (parsed $ OR power
# capacity), and a primary SEC source -- is the kind of drop most worth a
# human false-drop re-check at scale. A deterministic hash on raw_id (never
# random()) samples a small fraction of them into drop_spotcheck.jsonl so a
# rerun re-samples exactly the same rows.
# ---------------------------------------------------------------------------
def _has_placeable_magnitude(row: dict) -> bool:
    amt = classify_amount(row.get("amount_text"), row.get("amount_scale"))
    if amt["amount_bn"] is not None:
        return True
    qv = row.get("quantity_value")
    return (row.get("quantity_unit") or "").strip().upper() in _POWER_UNITS and qv is not None


def drop_has_strong_signals(row: dict) -> bool:
    """Named target AND a placeable magnitude AND a primary 'sec_' source."""
    return (bool(row.get("target_firm"))
            and _has_placeable_magnitude(row)
            and (row.get("source_type") or "").lower().startswith("sec_"))


def _spotcheck_bucket(raw_id: str) -> int:
    """Deterministic 0..999999 bucket for a raw_id (stable across reruns)."""
    return int(hashlib.sha1(f"dropspot|{raw_id}".encode()).hexdigest(), 16) % 1_000_000


def sample_drop_spotcheck(dropped: list, rate: float) -> list:
    """Deterministic hash-based sample (rate in [0,1]) of the strong-signal drops.
    Returns a list of copies, each tagged _spotcheck_reason. rate<=0 -> []."""
    if not rate or rate <= 0:
        return []
    threshold = rate * 1_000_000
    out = []
    for row in dropped:
        if not drop_has_strong_signals(row):
            continue
        if _spotcheck_bucket(str(row.get("raw_id") or "")) < threshold:
            r = dict(row)
            r["_spotcheck_reason"] = "strong_signal_drop"
            out.append(r)
    return out


def sort_rows(rows: list) -> list:
    rows.sort(key=lambda r: (r.get("target_firm") or "",
                             r.get("doc_date") or "",
                             r.get("instrument_class") or ""))
    return rows


# Observation column order. Stage 3 writes a separate audit table by raw_id;
# the Fact registry writes its own relational outputs downstream.
INSTANCE_COLUMNS = [
    "schema_version", "raw_id",
    "target_firm", "target_is_multi_entity", "target_parent", "target_parent_relation",
    "amount_text", "amount_scale", "currency", "amount_basis",
    "amount_bound_type_extracted", "amount_aggregation_level",
    "quantity_value", "quantity_unit", "temporal_context",
    "instrument_class", "debt_instrument_subtype", "method_summary",
    "object_name", "object_type", "object_evidence",
    "parent_object_name", "parent_object_type", "parent_object_relation",
    "parent_object_evidence", "counterparty",
    "economic_action", "capital_provider", "capital_recipient",
    "other_parties",
    "ai_infra_scope", "geo_scope", "claim_source", "claim_source_name",
    "deal_status", "deal_status_cue",
    "commitment_status", "commitment_status_cue", "conditionality", "condition_text",
    "summary", "verbatim_quote", "field_evidence", "evidence_cleared",
    "schema_defects",
    "amount_needs_review", "amount_review_reason",
    "audit_time_bad", "direction_fix_applied", "direction_fix_source",
    "direction_fix_reviewer",
    "audit_should_mismatch",
    "source_type", "doc_id", "source_path", "doc_date", "firm", "fiscal_period", "chunk_id",
]
