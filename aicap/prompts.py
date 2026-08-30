#!/usr/bin/env python3
"""System prompts for the extraction stages, with their versions and validators.

One prompt per LLM stage.  A prompt's sha1 hash is its version string, so
editing a prompt automatically gives that stage a fresh cache namespace and
never invalidates any other stage.

  * SCREEN  (Stage 1) -- short binary relevance screen.  Runs that feed a
                         pre-screened chunk set skip Stage 1 entirely
                         (seed_all_relevant); the prompt is kept here for
                         full-pool runs.
  * EXTRACT (Stage 2) -- extract + classify capital observations.
  * REVIEW  (Stage 3) -- pure auditor: eight substance checks, each graded
                         ok|caveat|bad|unsure, no overall verdict.

The two large prompts (EXTRACT, REVIEW) live in sibling .txt files at the
pipeline root so they are easy to edit; they are still hash-versioned.
"""
from __future__ import annotations
import json
import os
import re
from numbers import Real
from .config import (AMOUNT_AGGREGATION_LEVEL_SET, AMOUNT_BASIS_SET,
                     AMOUNT_BOUND_TYPE_SET, CLAIM_SOURCE_SET,
                     DEBT_INSTRUMENTS, DEBT_INSTRUMENT_SUBTYPE_SET,
                     INSTRUMENT_CLASS_SET, OBJECT_PARENT_TYPES, OBJECT_TYPE_SET,
                     PARENT_OBJECT_RELATION_SET, TARGET_PARENT_RELATION_SET,
                     is_news_source)
from .llm import prompt_hash, sha16

# The .txt prompt files sit at the pipeline root (the aicap package's parent).
_PROMPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), encoding="utf-8") as fh:
        return fh.read().strip() + "\n"


# ===========================================================================
# Stage 1 -- binary relevance screen (skipped for pre-screened chunk sets)
# ===========================================================================
SCREEN_PROMPT = """You are a stage-1 binary relevance screen for AI / data-center infrastructure capital. You get ONE excerpt; flag whether it shows someone taking on infra-capital obligations.

Output relevant=1 if and only if the excerpt contains at least one capital OBLIGATION / FINANCING COMMITMENT that is BOTH:
 (A) AI infra: data centers / compute / servers / GPUs / cloud capacity / power for compute -- or an infra-financing instrument tied to it (lease, build-to-suit, PPA / EAPA / offtake, equipment / supplier purchase or prepayment, compute / cloud / GPU capacity contract or reservation, SPV/JV/VIE, construction or project debt, guarantee / credit support / backstop of such obligations, government grant / subsidy for such infra, facility or land PP&E) -- or capital INTO the companies that build, run, or anchor it: M&A / acquisition of an AI-infra, cloud, chip, data-center, or AI-model company, or of their operating data-center assets / portfolios; an equity or venture investment into such a company -- AI model developers / AI labs / foundation-model companies COUNT (capital into an OpenAI / Anthropic-class lab is in scope); financing (debt or equity) RAISED or secured by such a company; AND
 (B) a capital position someone TAKES ON -- a commitment / obligation / exposure, or the INFLOW that creates one. Financing RAISED counts: a company securing debt or equity is taking on obligations. A government grant / subsidy AWARDED for such infra counts. A GUARANTEE issued counts: the guarantor takes contingent exposure. (Plain corporate debt counts only if its proceeds are explicitly for AI / data-center infra.) A utility's capex program counts only when data-center / AI demand is explicitly cited as a driver.
Scan EVERY sentence: excerpts are often digests of many unrelated headlines, or long narratives about something else. ONE qualifying sentence anywhere makes relevant=1, regardless of the excerpt's dominant topic.
Be HIGH-RECALL when unsure.

relevant=0 for: revenue / RPO / backlog; depreciation / buybacks / SBC / dividends / tax / generic narrative; boilerplate with NO obligation AND NO number.

Respond with ONLY JSON, no prose, no code fences:
{"relevant":0 or 1,"reason":"<=12 words"}"""

SCREEN_VERSION = "screen_" + prompt_hash(SCREEN_PROMPT)


# ===========================================================================
# Stage 2 -- capital-fact extraction + classification
# ===========================================================================
EXTRACT_PROMPT = _load("extract_prompt.txt")
EXTRACT_VERSION = "extract_" + prompt_hash(EXTRACT_PROMPT)


# ===========================================================================
# Stage 3 -- pure defensive review: 8 checks, per-check grade, no verdict
# ===========================================================================
REVIEW_PROMPT = _load("review_prompt.txt")
REVIEW_VERSION = "review_" + prompt_hash(REVIEW_PROMPT)

# The eight substance checks the auditor grades.  `anchor_direction` also
# audits that counterparty is the correct single other side; `time_ok` audits
# the free-text temporal_context field.
# WIRING GATE: these keys MUST byte-match the output JSON schema in
# review_prompt.txt.  If they drift, valid_review fails every row, every audit
# becomes _UNPARSABLE_AUDIT (all 'unsure'), and the audit-bad drop lane
# silently becomes a no-op.
REVIEW_CHECK_KEYS = ("in_scope", "anchor_direction", "amount_ok", "amount_basis_ok",
                     "instrument_ok", "object_ok", "ai_scope_ok", "time_ok")
REVIEW_ROLE_KEYS = ("economic_action", "expected_capital_provider",
                    "expected_capital_recipient", "expected_target_firm",
                    "expected_counterparty")


# ---------------------------------------------------------------------------
# User-message builders
# ---------------------------------------------------------------------------
# Excerpt cap on the rendered user message.  26,000 covers every chunk in
# both v8.1 corpora untruncated (max observed 25,929 chars): the chunker
# packs bodies to MAX_TOKENS*4 = 24,000 chars, but the transcript path
# prepends its ~600-char header AFTER packing and a single oversized speaker
# turn is kept whole, so 153 transcript chunks land above 24,000.  At the old
# 24,000 cap their tails were trimmed at render time; 139 were redundantly
# covered by the chunker's overlap, 14 (final call windows) lost their last
# 63-202 chars.  NOTE: this cap is NOT part of the LLM cache identity (which
# digests the chunk TEXT, not the rendered message) -- changing it against a
# warm cache would replay old renderings; change it only alongside a fresh
# cache or a prompt-version bump.
_MAX_EXCERPT_CHARS = 26000


def _firm_for_header(chunk: dict) -> str:
    firm = chunk.get("firm")
    if not firm:
        fm = chunk.get("firms_mentioned") or []
        firm = "/".join(fm) if fm else "(unresolved)"
    return firm


def build_user(chunk: dict) -> str:
    """Stage 1 / Stage 2 user message: provenance header + excerpt.
    SEC / transcript sources carry a FILER header (the filing entity is the
    default obligor).  News sources carry only FIRMS_MENTIONED: the obligor
    must be read from the sentence, never from the headline subject."""
    st = chunk.get("source_type") or ""
    if is_news_source(st):
        fm = chunk.get("firms_mentioned") or []
        who = f"FIRMS_MENTIONED: {', '.join(fm) if fm else '(none)'}"
    else:
        who = f"FILER: {_firm_for_header(chunk)}"
    head = (f"SOURCE: {chunk.get('source_type')} | {who} | "
            f"FISCAL_PERIOD: {chunk.get('fiscal_period')} | "
            f"DOC_DATE: {chunk.get('doc_date')} | "
            f"SECTION: {chunk.get('section_label')}")
    return head + '\n\nEXCERPT:\n"""\n' + (chunk.get("text") or "")[:_MAX_EXCERPT_CHARS] + '\n"""'


# The extracted fields the Stage-3 auditor sees and must verify, in fixed order.
_REVIEW_FIELDS = ["target_firm", "target_parent", "target_parent_relation",
                   "amount_text", "amount_scale", "currency", "amount_basis",
                   "amount_bound_type_extracted", "amount_aggregation_level",
                   "quantity_value", "quantity_unit",
                   "temporal_context",
                   "object_name", "object_type", "object_evidence",
                   "parent_object_name", "parent_object_type",
                   "parent_object_relation", "parent_object_evidence",
                   "counterparty", "economic_action", "capital_provider",
                   "capital_recipient", "other_parties",
                   "instrument_class", "ai_infra_scope", "geo_scope",
                   "debt_instrument_subtype", "claim_source", "claim_source_name",
                   "deal_status", "deal_status_cue",
                  "commitment_status", "commitment_status_cue",
                  "conditionality", "condition_text",
                  "method_summary", "verbatim_quote", "field_evidence", "summary"]


def _review_cand(obs: dict) -> dict:
    """The auditor-visible fields in _REVIEW_FIELDS order.  Shared by
    build_review_user and review_content_digest so the rendered message and
    the cache digest serialize byte-identical content."""
    return {k: obs.get(k) for k in _REVIEW_FIELDS}


def build_review_user(obs: dict, chunk_text: str) -> str:
    """Stage 3 user message: the source excerpt + the extracted instance."""
    head = (f"SOURCE: {obs.get('source_type')} | "
            f"TARGET(extracted): {obs.get('target_firm')} | "
            f"DOC_DATE: {obs.get('doc_date')}")
    return (head + '\n\nEXCERPT:\n"""\n' + (chunk_text or "")[:_MAX_EXCERPT_CHARS] + '\n"""'
            + "\n\nEXTRACTED INSTANCE (audit this):\n"
            + json.dumps(_review_cand(obs), ensure_ascii=False, indent=1))


def review_content_digest(obs: dict) -> str:
    """sha16 over the auditor-visible fields, serialized exactly as
    build_review_user renders them.  Folded into the Stage-3 review cache id
    so any upstream rewrite of an auditor-visible field (e.g. the hostable
    screen nulling amount_text) yields a fresh cache identity and a fresh
    verdict; byte-identical content keeps the cached verdict.  The excerpt is
    not in the digest -- it is fixed per raw_id (chunk_id is part of raw_id)."""
    return sha16(json.dumps(_review_cand(obs), ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# Schema validators (passed to LLM.call_json).  Stage 2 is intentionally strict:
# a response is complete only when every observation satisfies the versioned
# field contract and controlled vocabularies. Semantic disputes still belong
# to Stage 3; malformed/missing fields do not.
# ---------------------------------------------------------------------------
def valid_screen(obj) -> bool:
    return isinstance(obj, dict) and obj.get("relevant") in (0, 1, True, False)


# Evidence grounding: every non-null evidence span must be real content of
# THIS observation's verbatim_quote.  The comparison collapses whitespace and
# casefolds both sides, and checks each " [...] " splice part separately --
# the same normalization philosophy the Stage-3 verbatim check applies
# (_vnorm): differences of newlines, runs of spaces, or letter case cannot
# fail a span whose words genuinely come from the quote, while an invented
# or out-of-quote span still fails.  (Pilot wave 1: byte-exact matching
# quarantined whole chunks over a newline inside the span.)
_GROUND_WS_RE = re.compile(r"\s+")


def _ground_norm(text) -> str:
    return _GROUND_WS_RE.sub(" ", str(text)).strip().casefold()


def _span_grounded(span, quote_norm: str) -> bool:
    for part in str(span).split(" [...] "):
        p = _ground_norm(part)
        if p and p not in quote_norm:
            return False
    return True


# ---------------------------------------------------------------------------
# The former strict extraction contract, restructured (2026-08-28 policy
# pivot, user decision): the TRANSPORT validator is minimal -- anything
# structurally parseable is admitted, so the repair lane fires only for
# genuinely unparsable JSON and batch fallbacks drop to ~zero.  The full
# contract lives on as the per-observation DEFECT RECORDER below
# (observation_defects): every violation is detected deterministically in
# code, defaulted/cleared at assembly, recorded on the row
# (schema_defects), rate-reported in run stats for the paper, and routed
# review_required at the quality gate so defective observations never
# become clean canonical evidence.  Nothing is repaired by re-prompting.
# ---------------------------------------------------------------------------
REQUIRED_OBSERVATION_KEYS = frozenset({
    "target_firm", "target_parent", "target_parent_relation", "amount_text",
    "amount_scale", "currency", "amount_basis", "amount_bound_type",
    "amount_aggregation_level", "quantity_value", "quantity_unit",
    "temporal_context", "object_name", "object_type", "object_evidence",
    "parent_object_name", "parent_object_type", "parent_object_relation",
    "parent_object_evidence", "counterparty", "economic_action",
    "capital_provider", "capital_recipient", "other_parties",
    "instrument_class", "debt_instrument_subtype", "claim_source",
    "claim_source_name", "ai_infra_scope", "geo_scope", "deal_status",
    "deal_status_cue", "commitment_status", "commitment_status_cue",
    "conditionality", "condition_text", "method_summary", "verbatim_quote",
    "field_evidence", "summary",
})
EVIDENCE_KEYS = frozenset({
    "target", "amount_or_capacity", "amount_basis", "amount_bound_type",
    "amount_aggregation_level", "instrument", "debt_subtype", "counterparty",
    "object", "parent_object", "claim_source", "ai_nexus",
    "period_or_horizon", "deal_status", "commitment_status", "parent",
})
OBSERVATION_ENUM_FIELDS = {
    "target_parent_relation": TARGET_PARENT_RELATION_SET,
    "amount_scale": {"trillion", "billion", "million", "thousand", "crore",
                     "lakh", "full_units", "none"},
    "amount_basis": AMOUNT_BASIS_SET,
    "amount_bound_type": AMOUNT_BOUND_TYPE_SET,
    "amount_aggregation_level": AMOUNT_AGGREGATION_LEVEL_SET,
    "economic_action": {"financing", "spending", "acquisition", "guarantee", "unclear"},
    "instrument_class": INSTRUMENT_CLASS_SET,
    "claim_source": CLAIM_SOURCE_SET,
    "ai_infra_scope": {"entire_amount", "mostly_infra", "partial_unallocated", "unclear"},
    "geo_scope": {"us_only", "includes_non_us", "unknown"},
    "deal_status": {"new_commitment", "update", "amendment", "periodic_disclosure",
                    "current_snapshot", "repeat_reference", "unclear"},
    "commitment_status": {"terminated", "cancelled", "delayed_or_paused",
                          "operational", "under_construction", "contracted",
                          "approved", "announced", "planned", "proposed", "unclear"},
    "conditionality": {"unconditional", "conditional", "unclear"},
}
_NULLABLE_STRING_FIELDS = frozenset({
    "target_firm", "amount_text", "quantity_unit", "temporal_context",
    "object_name", "object_evidence", "parent_object_name",
    "parent_object_evidence", "counterparty", "capital_provider",
    "capital_recipient", "debt_instrument_subtype", "claim_source_name",
    "deal_status_cue", "commitment_status_cue", "condition_text",
    "method_summary",
})
_MAX_DEFECTS_PER_OBSERVATION = 12

# Keys the PIPELINE adds to an observation after extraction (provenance
# enrichment in run_extract, raw_id in collect_raw_instances, salvage tags).
# The defect scanner must judge only what the MODEL emitted, so these are
# never "unexpected"; underscore-prefixed keys are internal by convention.
ENRICHMENT_KEYS = frozenset({
    "firm", "source_type", "fiscal_period", "doc_date", "section_label",
    "chunk_id", "doc_id", "char_start", "char_end", "source_path",
    "firms_mentioned", "raw_id",
})


def valid_extract(obj) -> bool:
    """TRANSPORT-MINIMAL validity: a dict with an `observations` list whose
    items are dicts.  Everything else is a recorded DEFECT, never a
    rejection -- see observation_defects and the module comment above."""
    if not isinstance(obj, dict) or not isinstance(obj.get("observations"), list):
        return False
    return all(isinstance(o, dict) for o in obj["observations"])


def observation_defects(o) -> list[str]:
    """Deterministic defect scan of ONE observation against the full v2
    contract.  Returns compact, stable defect codes (paper-tally-friendly);
    empty list = fully contract-clean.  Consumed by assemble._prep_row
    (defaults/clears + schema_defects column + run-stats rates)."""
    if not isinstance(o, dict):
        return ["not_a_dict"]
    d = []
    missing = REQUIRED_OBSERVATION_KEYS - set(o)
    extra = {k for k in set(o) - REQUIRED_OBSERVATION_KEYS - ENRICHMENT_KEYS
             if not str(k).startswith("_")}
    if missing:
        d.append("missing_keys:" + ",".join(sorted(missing)[:6]))
    if extra:
        d.append("unexpected_keys:" + ",".join(sorted(str(k)[:30] for k in extra)[:6]))
    if "verbatim_quote" in o and (not isinstance(o.get("verbatim_quote"), str)
                                  or not o["verbatim_quote"].strip()):
        d.append("empty_verbatim_quote")
    if "summary" in o and (not isinstance(o.get("summary"), str)
                           or not o["summary"].strip()):
        d.append("empty_summary")
    elif isinstance(o.get("summary"), str) and len(o["summary"].split()) > 70:
        d.append("summary_overlength")
    if isinstance(o.get("method_summary"), str) \
            and len(o["method_summary"].split()) > 12:
        d.append("method_summary_overlength")
    for field, values in OBSERVATION_ENUM_FIELDS.items():
        if field in o and o.get(field) not in values:
            d.append("invalid_enum:" + field)
    for field in _NULLABLE_STRING_FIELDS:
        if field in o and o.get(field) is not None \
                and not isinstance(o.get(field), str):
            d.append("nonstring:" + field)
    if "currency" in o and (not isinstance(o.get("currency"), str)
                            or not o["currency"].strip()):
        d.append("invalid_currency")
    if "other_parties" in o and (not isinstance(o.get("other_parties"), list)
                                 or not all(isinstance(x, str)
                                            for x in o["other_parties"])):
        d.append("other_parties_not_list")
    if o.get("quantity_value") is not None and (
            isinstance(o.get("quantity_value"), bool)
            or not isinstance(o.get("quantity_value"), Real)):
        d.append("invalid_quantity_value")
    if not (o.get("target_parent") is None
            or isinstance(o.get("target_parent"), str)
            or (isinstance(o.get("target_parent"), list)
                and all(isinstance(x, str) for x in o["target_parent"]))):
        d.append("invalid_target_parent")
    # object / parent joint rules
    if o.get("object_name") is None:
        if any(o.get(k) is not None for k in ("object_type", "object_evidence",
                                              "parent_object_name",
                                              "parent_object_type",
                                              "parent_object_relation",
                                              "parent_object_evidence")):
            d.append("object_metadata_without_name")
    elif o.get("object_type") is not None \
            and o.get("object_type") not in OBJECT_TYPE_SET:
        d.append("invalid_object_type")
    if o.get("parent_object_name") is not None:
        if o.get("parent_object_type") is not None \
                and o.get("parent_object_type") not in OBJECT_TYPE_SET:
            d.append("invalid_parent_object_type")
        if o.get("parent_object_relation") is not None \
                and o.get("parent_object_relation") not in PARENT_OBJECT_RELATION_SET:
            d.append("invalid_parent_relation")
        if o.get("parent_object_name") == o.get("object_name"):
            d.append("parent_equals_object")
        if (o.get("parent_object_relation") == "component_of"
                and o.get("parent_object_type") is not None
                and o.get("object_type") is not None
                and o.get("parent_object_type") not in
                OBJECT_PARENT_TYPES.get(o.get("object_type"), set())):
            d.append("parent_type_incompatible:" + str(o.get("object_type"))
                     + "<-" + str(o.get("parent_object_type")))
    subtype = o.get("debt_instrument_subtype")
    if o.get("instrument_class") in DEBT_INSTRUMENTS:
        if subtype not in DEBT_INSTRUMENT_SUBTYPE_SET:
            d.append("debt_subtype_missing_or_invalid")
    elif subtype is not None:
        d.append("nondebt_subtype_present")
    if "field_evidence" in o:
        ev = o.get("field_evidence")
        if not isinstance(ev, dict):
            d.append("field_evidence_not_dict")
        else:
            ev_extra = set(ev) - EVIDENCE_KEYS
            ev_missing = EVIDENCE_KEYS - set(ev)
            if ev_extra:
                d.append("evidence_unknown_keys:"
                         + ",".join(sorted(str(k)[:30] for k in ev_extra)[:5]))
            if ev_missing:
                d.append("evidence_missing_keys:"
                         + ",".join(sorted(ev_missing)[:5]))
            if any(v is not None and not isinstance(v, str)
                   for v in ev.values()):
                d.append("evidence_nonstring_value")
    return d[:_MAX_DEFECTS_PER_OBSERVATION]


def extract_invalid_reason(obj) -> str:
    """Explain a transport-level rejection (now shape-only) or, for
    diagnostics, the first observation's defects.  Kept for the parse-repair
    message hook (valid_extract.explain) and tooling."""
    if not isinstance(obj, dict) or not isinstance(obj.get("observations"), list):
        return 'top level must be exactly {"observations": [...]}'
    for i, o in enumerate(obj["observations"]):
        defects = observation_defects(o)
        if defects:
            return "observations[" + str(i) + "]: " + "; ".join(defects)
    return "no defects detected"


def valid_review(obj) -> bool:
    """Per-check assessment only (no verdict): every check in
    REVIEW_CHECK_KEYS present and graded ok|caveat|bad|unsure, the role
    reconstruction present, plus an optional short note."""
    if not isinstance(obj, dict) or set(obj) != {"role_resolution", "checks", "note"}:
        return False
    roles = obj.get("role_resolution")
    if not isinstance(roles, dict) or set(roles) != set(REVIEW_ROLE_KEYS):
        return False
    if roles.get("economic_action") not in ("financing", "spending", "acquisition",
                                             "guarantee", "unclear"):
        return False
    for key in REVIEW_ROLE_KEYS[1:]:
        if roles.get(key) is not None and not isinstance(roles.get(key), str):
            return False
    checks = obj.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REVIEW_CHECK_KEYS):
        return False
    for k in REVIEW_CHECK_KEYS:
        if checks.get(k) not in ("ok", "caveat", "bad", "unsure"):
            return False
    note = obj.get("note")
    if note is not None and not isinstance(note, str):
        return False
    return True


# Attach the advisory explainer so the transport's repair retry can name the
# failing rule (llm._parse_valid looks for `validate.explain`).
valid_extract.explain = extract_invalid_reason
