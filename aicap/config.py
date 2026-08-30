#!/usr/bin/env python3
"""Domain configuration: instrument taxonomy, fiscal calendars, source
preference, and the model registry.
"""
from __future__ import annotations
import calendar
import datetime as dt
import os
from typing import Optional

# One version for every persisted artifact produced by the aggregation-ready
# redesign.  Prompt hashes still version individual LLM stages, while this
# value versions the cross-stage data contract.
SCHEMA_VERSION = "fact_registry_v2"

# Fraction of STRONG-SIGNAL gate/audit drops sampled into drop_spotcheck.jsonl
# for a human false-drop re-check. Deterministic hash-on-raw_id sampling
# (never random()), so a rerun samples exactly the same rows. 0 disables.
DROP_SPOTCHECK_RATE = float(os.environ.get("AICAP_DROP_SPOTCHECK_RATE", "0.02"))

# ---------------------------------------------------------------------------
# Models.  Two transport families are registered:
#   anthropic      Claude via the Anthropic SDK (Messages + Message Batches)
#   openai_compat  Qwen via the DashScope OpenAI-compatible endpoints
#                  (realtime + the half-price Batch Chat endpoint)
# ---------------------------------------------------------------------------
# Each short key -> a spec dict. Fields:
#   provider  transport family (see aicap.llm).
#   id        the wire model-id string. ResultCache keys are (item, stage,
#             prompt_version, model_id) -- this string is load-bearing: change
#             it and every cached verdict for that model becomes unreachable.
#             Different models' verdicts coexist in one cache without
#             collision, so switching profiles never destroys old results.
#   max_out   the model's maximum output tokens. CallConfig.max_tokens=None
#             resolves to this ceiling; the stop_reason=="max_tokens" backstop
#             still fires so truncation can never be silent.
#   in/out    price per 1M input / output tokens, in `currency` (telemetry
#             only; Batch Chat bills these at 50%).
#   base_url / batch_base_url / api_key_env   (openai_compat only) endpoint
#             URLs and the env var holding the API key; base_url_env may
#             override base_url.
# Prefer the model_id_of()/model_spec() accessors over indexing MODELS.
MODELS = {
    "haiku-4.5":  {"provider": "anthropic", "id": "claude-haiku-4-5-20251001",
                   "in": 1.00, "out": 5.00, "currency": "USD", "max_out": 64000},
    # 2026-08-28: the anthropic profile's workhorse is Opus 5 (same $5/$25
    # list price and 128K output ceiling as 4.8).  API differences the
    # transport already absorbs: sampling params are removed (temperature is
    # never sent -- pre-seeded in AnthropicProvider), and thinking is ON by
    # default (adaptive); thinking blocks are skipped by the text assembler
    # but their tokens bill as output, so expect somewhat higher out_tokens
    # per call than 4.8 at the same list price.
    "opus-5":     {"provider": "anthropic", "id": "claude-opus-5",
                   "in": 5.00, "out": 25.00, "currency": "USD", "max_out": 128000},
    # kept for cache addressability and switch-back (cached verdicts are
    # keyed by model_id; removing the entry would strand them)
    "opus-4.8":   {"provider": "anthropic", "id": "claude-opus-4-8",
                   "in": 5.00, "out": 25.00, "currency": "USD", "max_out": 128000},
    "qwen3.6-flash": {
        "provider": "openai_compat", "id": "qwen3.6-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "batch_base_url": "https://batch.dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "DASHSCOPE_BASE_URL", "api_key_env": "DASHSCOPE_API_KEY",
        "in": 1.2, "out": 7.2, "currency": "CNY", "max_out": 65536},
    "qwen3.7-max": {
        "provider": "openai_compat", "id": "qwen3.7-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "batch_base_url": "https://batch.dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "DASHSCOPE_BASE_URL", "api_key_env": "DASHSCOPE_API_KEY",
        "in": 6.0, "out": 18.0, "currency": "CNY", "max_out": 131072},
    # in/out are PROVISIONAL -- copied from 3.7-max pending the published rate
    # card; they feed cost telemetry only, never routing.
    "qwen3.8-max": {
        "provider": "openai_compat", "id": "qwen3.8-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "batch_base_url": "https://batch.dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "DASHSCOPE_BASE_URL", "api_key_env": "DASHSCOPE_API_KEY",
        "in": 6.0, "out": 18.0, "currency": "CNY", "max_out": 131072},
}

# ---------------------------------------------------------------------------
# Per-stage model assignment, switchable as ONE profile:
#     AICAP_MODEL_PROFILE = qwen (default) | anthropic
# Individual stage overrides win over the profile when set:
#     AICAP_SCREEN_MODEL / AICAP_EXTRACT_MODEL / AICAP_REVIEW_MODEL /
#     AICAP_NORM_MODEL / AICAP_REG_MODEL
# Switching profiles changes the run's instrument: every LLM verdict is keyed
# by model_id, so the first run under a new profile re-bills every LLM stage
# while the old profile's cache rows stay intact for switching back.
# ---------------------------------------------------------------------------
_PROFILES = {
    "anthropic": {"screen": "haiku-4.5", "extract": "opus-5",
                  "review": "opus-5", "norm": "opus-5", "reg": "opus-5"},
    "qwen": {"screen": "qwen3.6-flash", "extract": "qwen3.7-max",
             "review": "qwen3.7-max", "norm": "qwen3.7-max",
             "reg": "qwen3.7-max"},
}
MODEL_PROFILE = os.environ.get("AICAP_MODEL_PROFILE", "qwen")
if MODEL_PROFILE not in _PROFILES:
    raise KeyError(f"unknown AICAP_MODEL_PROFILE {MODEL_PROFILE!r}; "
                   f"known: {sorted(_PROFILES)}")


def _stage_model(stage: str) -> str:
    key = os.environ.get(f"AICAP_{stage.upper()}_MODEL") \
        or _PROFILES[MODEL_PROFILE][stage]
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r} for stage {stage!r}; "
                       f"known: {sorted(MODELS)}")
    return key


SCREEN_MODEL = _stage_model("screen")     # Stage 1 relevance screen
EXTRACT_MODEL = _stage_model("extract")   # Stage 2 extract + classify
REVIEW_MODEL = _stage_model("review")     # Stage 3 pure auditor
NORM_MODEL = _stage_model("norm")         # time-claim extraction + grading
REG_MODEL = _stage_model("reg")           # registry Judge (amounts/entities/facts)


def model_id_of(key: str) -> str:
    """The wire model-id string for a short key (== the cache's model_id field).
    Raises KeyError on an unknown key: a typo'd model must not silently fall
    back to a different, cache-colliding id."""
    return MODELS[key]["id"]


def model_spec(key: str) -> dict:
    """Full spec for a MODELS short key, as a fresh dict; `provider` is
    guaranteed present."""
    spec = dict(MODELS[key])
    spec.setdefault("provider", "anthropic")
    return spec


def stage_models() -> dict:
    """Per-stage model provenance for run reports: each LLM stage's short key,
    resolved wire id, and provider (the run's instrument-version record)."""
    keys = {"screen": SCREEN_MODEL, "extract": EXTRACT_MODEL,
            "review": REVIEW_MODEL, "norm": NORM_MODEL,
            "registry_judge": REG_MODEL}
    return {stage: {"model_key": key, "model_id": model_id_of(key),
                    "provider": model_spec(key)["provider"]}
            for stage, key in keys.items()}

# ===========================================================================
# Firm fiscal calendars: every corpus ticker + the major counterparties of the
# AI capital network, keyed on a normalized name so canonical registry forms
# ("Meta Platforms, Inc.", "NVIDIA Corporation") resolve without enumerating
# every suffix variant. Values are FY START months (FY label = ENDING year for
# non-January starts: Microsoft FY2025 = 2024-07-01..2025-06-30).
# A firm absent from this table has an UNKNOWN calendar: the time layer then
# keeps FY labels unresolved rather than assuming a January year.
# ===========================================================================
FISCAL_YEAR_START_MONTH = {
    # corpus tickers (19; universe v2 added CORZ + GLXY, both calendar-year filers)
    "Microsoft": 7, "Oracle": 6, "Amazon": 1, "Alphabet": 1, "Meta": 1,
    "Applied Digital": 6, "IREN": 7, "Blackstone": 1, "Constellation Energy": 1,
    "Cipher Mining": 1, "CoreWeave": 1, "Nebius": 1, "NRG Energy": 1,
    "Blue Owl Capital": 1, "Talen Energy": 1, "Vistra": 1, "TeraWulf": 1,
    "Core Scientific": 1, "Galaxy Digital": 1,
    # major AI-capital counterparties with well-documented calendars
    "Nvidia": 2, "SoftBank Group": 4, "Arm Holdings": 4, "Dell Technologies": 2,
    "Super Micro Computer": 7, "Apple": 10, "Broadcom": 11, "Micron Technology": 9,
    "Salesforce": 2, "Cisco Systems": 8, "Hewlett Packard Enterprise": 11,
    "Intel": 1, "Advanced Micro Devices": 1, "Taiwan Semiconductor Manufacturing": 1,
    "Equinix": 1, "Digital Realty": 1, "Alibaba Group": 4, "Samsung Electronics": 1,
    "SK Hynix": 1,
}
# 52/53-week filers: month-boundary bounds are a <=6-day approximation of the
# true week-based period ends. Honest enough for quarter bucketing; surfaced in
# fiscal_calendar_description so the time LLM and reviewers know.
WEEK_CALENDAR_APPROX = {"Nvidia", "Dell Technologies", "Apple", "Broadcom",
                        "Micron Technology", "Cisco Systems", "Intel",
                        "Advanced Micro Devices"}

# Genuinely different names (the normalized-key lookup already absorbs legal
# suffixes, punctuation, and case).
FISCAL_CALENDAR_ALIASES = {
    "Amazon.com": "Amazon", "Amazon.com, Inc.": "Amazon",
    "Amazon Web Services": "Amazon", "AWS": "Amazon",
    "CORZ": "Core Scientific",
    "GLXY": "Galaxy Digital", "Galaxy Digital Holdings": "Galaxy Digital",
    "Google": "Alphabet", "Google LLC": "Alphabet",
    "Meta Platforms": "Meta", "Facebook": "Meta",
    "Iris Energy": "IREN",
    "NVIDIA": "Nvidia",
    "SoftBank": "SoftBank Group", "Arm": "Arm Holdings",
    "Dell": "Dell Technologies",
    "Supermicro": "Super Micro Computer", "Super Micro": "Super Micro Computer",
    "AMD": "Advanced Micro Devices",
    "TSMC": "Taiwan Semiconductor Manufacturing",
    "Taiwan Semiconductor": "Taiwan Semiconductor Manufacturing",
    "HPE": "Hewlett Packard Enterprise",
    "Cisco": "Cisco Systems",
    "Micron": "Micron Technology",
    "Alibaba": "Alibaba Group", "Alibaba Group Holding": "Alibaba Group",
    "Samsung": "Samsung Electronics",
    "NRG": "NRG Energy", "Blue Owl": "Blue Owl Capital",
    "Talen": "Talen Energy", "Constellation": "Constellation Energy",
    "Cipher": "Cipher Mining",
    "The Blackstone Group": "Blackstone",
    "Vistra Energy": "Vistra", "Vistra Corp": "Vistra",
    "Nebius Group": "Nebius",
    "Salesforce.com": "Salesforce",
}

_LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "llc", "lp", "llp", "co", "company", "nv", "n.v", "sa", "s.a", "ag",
    "holdings", "holding",
}


def _fiscal_key(name: str) -> str:
    """Lower-case, punctuation-free, legal-suffix-stripped lookup key."""
    toks = [t for t in
            (tok.strip(".").lower() for tok in
             str(name).replace(",", " ").replace("&", " ").split())
            if t and t not in _LEGAL_SUFFIX_TOKENS]
    return " ".join(toks)


def _build_fiscal_index() -> dict:
    index = {}
    for display, month in FISCAL_YEAR_START_MONTH.items():
        index[_fiscal_key(display)] = display
    for alias, display in FISCAL_CALENDAR_ALIASES.items():
        index.setdefault(_fiscal_key(alias), display)
    return index


_FISCAL_KEY_INDEX = _build_fiscal_index()


def fiscal_calendar_firm(firm: Optional[str]) -> Optional[str]:
    """Resolve a surface/canonical name to its calendar display name, or None."""
    name = (firm or "").strip()
    if not name:
        return None
    display = FISCAL_CALENDAR_ALIASES.get(name, name)
    if display in FISCAL_YEAR_START_MONTH:
        return display
    return _FISCAL_KEY_INDEX.get(_fiscal_key(name))


def known_fiscal_start_month(firm: Optional[str]) -> Optional[int]:
    """Return a documented firm calendar; never apply a calendar-year default."""
    display = fiscal_calendar_firm(firm)
    return FISCAL_YEAR_START_MONTH.get(display) if display else None


def fiscal_calendar_description(firm: Optional[str]) -> str:
    display = fiscal_calendar_firm(firm)
    start = FISCAL_YEAR_START_MONTH.get(display) if display else None
    if start is None:
        return "unknown; do not map FY labels to calendar dates"
    end = 12 if start == 1 else start - 1
    approx = " (52/53-week filer; month-boundary approximation)" if display in WEEK_CALENDAR_APPROX else ""
    return f"known: fiscal year starts month {start:02d} and ends month {end:02d}{approx}"


def fiscal_year_bounds(firm: Optional[str], fiscal_year: int) -> Optional[tuple[str, str]]:
    """Map FYyyyy to full dates only for firms with explicit calendar metadata."""
    start_month = known_fiscal_start_month(firm)
    if start_month is None:
        return None
    start_year = fiscal_year if start_month == 1 else fiscal_year - 1
    end_month = 12 if start_month == 1 else start_month - 1
    end_year = fiscal_year
    end_day = calendar.monthrange(end_year, end_month)[1]
    return (dt.date(start_year, start_month, 1).isoformat(),
            dt.date(end_year, end_month, end_day).isoformat())


def fiscal_quarter_bounds(firm: Optional[str], fiscal_year: int,
                          fiscal_q: int) -> Optional[tuple[str, str]]:
    """Inclusive date bounds of a firm's fiscal quarter (FY label = ending year).
    No silent calendar default -- an unknown firm returns None."""
    start_month = known_fiscal_start_month(firm)
    if start_month is None or fiscal_q not in (1, 2, 3, 4):
        return None
    start_year = fiscal_year if start_month == 1 else fiscal_year - 1
    m0 = start_month + 3 * (fiscal_q - 1)
    y0 = start_year + (m0 - 1) // 12
    m0 = (m0 - 1) % 12 + 1
    m1 = m0 + 2
    y1 = y0 + (m1 - 1) // 12
    m1 = (m1 - 1) % 12 + 1
    end_day = calendar.monthrange(y1, m1)[1]
    return (dt.date(y0, m0, 1).isoformat(), dt.date(y1, m1, end_day).isoformat())


# ===========================================================================
# PRIMARY AXIS -- instrument_class. EXACTLY ONE per observation: the most
# specific capital arrangement by PRECEDENCE (first that genuinely applies).
# The label/blurb prose documents the taxonomy for humans; the model-facing
# wording lives in extract_prompt.txt.
# ===========================================================================
INSTRUMENT_CLASSES = {
    "jv_spv": {"label": "Stake / commitment into a JV / SPV / VIE",
               "blurb": "The figure IS a measured stake / membership interest / capital "
                        "commitment into a JV / SPV / VIE / 'the Venture'."},
    "m_and_a": {"label": "M&A / control acquisition",
                "blurb": "Acquisition of CONTROL of a company / business / platform with a "
                         "stated AI/DC nexus. target=ACQUIRER; the acquired company or business "
                         "is represented by the typed object fields; "
                         "counterparty=SELLER."},
    "equity_contribution": {"label": "Equity / preferred-equity (minority / non-control)",
                            "blurb": "Equity / preferred / PIPE / strategic minority equity INTO a "
                                     "DC/compute party; NOT a measured vehicle stake, NOT control."},
    "project_debt": {"label": "Project / construction / SPV-level debt",
                     "blurb": "Construction / project-finance / ABS debt against a specific "
                              "facility or SPV (project/facility/SPV-level or non-recourse)."},
    "corporate_debt": {"label": "Issuer-level corporate debt",
                       "blurb": "Issuer-level notes / bonds / CP / term loan / revolver -- ONLY if "
                                "proceeds explicitly for AI/DC. Convertibles = corporate_debt."},
    "guarantee": {"label": "Guarantee / credit support",
                  "blurb": "Residual-value / completion guarantee, indemnity, letter of credit, "
                           "surety / backstop. target=GUARANTOR; counterparty=beneficiary."},
    "ppa_power": {"label": "Power purchase / offtake (PPA / EAPA)",
                  "blurb": "Power purchase / offtake / capacity agreement; the BUYER / offtaker "
                           "commits (generator/utility is the counterparty)."},
    "lease": {"label": "Lease (operating / finance / build-to-suit)",
              "blurb": "Operating / finance / build-to-suit / not-yet-commenced lease or "
                       "sale-leaseback -- even when the asset sits in PP&E."},
    "supplier_purchase": {"label": "Supplier / equipment purchase commitment",
                          "blurb": "Server / GPU / chip / equipment / component purchase or prepay "
                                    "commitment (the buyer commits)."},
    "compute_capacity_contract": {
        "label": "Compute / cloud / hosting capacity contract",
        "blurb": "A quantified purchase or reservation of compute, cloud, GPU, "
                 "hosting, or data-center service capacity; target is the buyer."},
    "asset_acquisition": {
        "label": "Operating infrastructure asset acquisition",
        "blurb": "Purchase of an operating facility, physical infrastructure asset, "
                 "or asset portfolio without acquiring control of a company/business."},
    "mixed_financing": {
        "label": "Inseparable mixed debt-and-equity financing",
        "blurb": "One quantified financing total explicitly combines debt and equity "
                 "and the source does not allocate the components."},
    "grant_subsidy": {
        "label": "Non-repayable grant / subsidy",
        "blurb": "A quantified non-repayable public award, grant, or subsidy provided "
                 "to the target for in-scope infrastructure."},
    "general_capex": {"label": "General capex / physical-asset exposure",
                      "blurb": "Catch-all capex / physical-asset exposure for AI/DC when no more "
                               "specific instrument applies: PP&E / CIP balance, period / guidance "
                               "capex flow, self-built capacity, or purchase of an existing asset "
                               "(land / building / plant)."},
    "other": {"label": "Other in-scope AI-infra obligation",
              "blurb": "A genuine obligation that is clearly none of the above; describe in "
                       "method_summary."},
    "unsure": {"label": "Unsure (evidence insufficient to classify)",
               "blurb": "Real obligation, class not confidently assignable."},
}
INSTRUMENT_CLASS_SET = set(INSTRUMENT_CLASSES)


def canonical_instrument_class(value: Optional[str]) -> Optional[str]:
    """Keep only a known instrument_class (lower-cased); else None."""
    if not value:
        return None
    k = str(value).strip().lower()
    return k if k in INSTRUMENT_CLASS_SET else None


# ===========================================================================
# CAPITAL-FLOW DIRECTION -- derived from instrument_class. Financing pulls
# capital IN to the target (source=counterparty, dest=target); spending pushes
# capital OUT from the target (source=target, dest=counterparty); guarantee is
# a contingent support edge target->counterparty. Entity ids are resolved by
# the registry stage; this only classifies the polarity.
# ===========================================================================
FINANCING_INSTRUMENTS = {"equity_contribution", "jv_spv", "project_debt",
                         "corporate_debt", "mixed_financing", "grant_subsidy"}
SPENDING_INSTRUMENTS = {"supplier_purchase", "lease",
                        "ppa_power", "compute_capacity_contract",
                        "general_capex", "m_and_a", "asset_acquisition"}


def flow_direction_of_instrument(instrument_class: Optional[str]) -> str:
    """Capital-flow polarity from instrument_class.
      inflow_to_target     -- financing: counterparty funds target (edge cp -> target)
      outflow_from_target  -- spending: target pays counterparty (edge target -> cp)
      contingent_support   -- guarantee: guarantor(target) -> beneficiary(cp), contingent
      unclear              -- other / unsure"""
    ic = (instrument_class or "").strip().lower()
    if ic in FINANCING_INSTRUMENTS:
        return "inflow_to_target"
    if ic in SPENDING_INSTRUMENTS:
        return "outflow_from_target"
    if ic == "guarantee":
        return "contingent_support"
    return "unclear"


# ===========================================================================
# SECONDARY AXES -- target_parent_relation / amount_basis.
# (deal_status and commitment_status are carried through from Stage 2 and are
# not re-validated here.)
# ===========================================================================

# target_parent_relation: na iff target_parent is null; the parent itself may
# be model-inferred.
TARGET_PARENT_RELATION_VALUES = ["na", "subsidiary", "spv_jv_vehicle",
                                 "brand_division", "unclear"]
TARGET_PARENT_RELATION_SET = set(TARGET_PARENT_RELATION_VALUES)

# The closed amount_basis vocabulary (byte-matches extract_prompt.txt).
# Stage-2 output is coerced against this set at assembly: an off-vocabulary
# basis becomes "unknown" + a review flag, so no invalid enum can flow into
# normalized output or metric keys.
AMOUNT_BASIS_VALUES = [
    "valuation", "carrying_value", "recognized_balance",
    "acquisition_consideration", "grant_subsidy", "financing_total",
    "equity_capital", "debt_principal", "facility_commitment",
    "contract_value", "undiscounted_commitment", "capex_cash_flow",
    "headline_project_cost", "investment_program_amount",
    "deposit_or_prepayment", "authorized_amount", "max_exposure",
    "capacity_only", "other", "unknown",
]
AMOUNT_BASIS_SET = set(AMOUNT_BASIS_VALUES)

AMOUNT_BOUND_TYPE_VALUES = ["exact", "approx", "min", "max", "range", "unclear"]
AMOUNT_BOUND_TYPE_SET = set(AMOUNT_BOUND_TYPE_VALUES)

AMOUNT_AGGREGATION_LEVEL_VALUES = [
    "object_total", "component_amount", "tranche", "period_flow",
    "cumulative_to_date", "incremental_change", "remaining_balance",
    "unit_rate", "unknown",
]
AMOUNT_AGGREGATION_LEVEL_SET = set(AMOUNT_AGGREGATION_LEVEL_VALUES)

CLAIM_SOURCE_VALUES = [
    "legal_or_filed_record", "deal_party_disclosure",
    "government_or_regulator", "named_external_source", "anonymous_source",
    "unattributed_media_report", "derived_from_reported_inputs", "unknown",
]
CLAIM_SOURCE_SET = set(CLAIM_SOURCE_VALUES)

DEBT_INSTRUMENT_SUBTYPE_VALUES = [
    "bond_or_note", "commercial_paper", "term_loan",
    "revolver_or_credit_facility", "securitization",
    "equipment_or_vendor_finance", "convertible_debt",
    "bridge_or_acquisition_finance", "government_backed_loan", "other", "unknown",
]
DEBT_INSTRUMENT_SUBTYPE_SET = set(DEBT_INSTRUMENT_SUBTYPE_VALUES)
DEBT_INSTRUMENTS = {"project_debt", "corporate_debt"}

OBJECT_TYPE_VALUES = [
    "data_center_project", "campus", "facility", "phase", "building",
    "power_project", "acquisition_company", "acquisition_business",
    "acquisition_asset_portfolio", "program",
]
OBJECT_TYPE_SET = set(OBJECT_TYPE_VALUES)

# Only containment is represented as an object parent.  Service, financing,
# ownership, and power-supply relationships are not parent edges.
PARENT_OBJECT_RELATION_VALUES = ["component_of", "unclear"]
PARENT_OBJECT_RELATION_SET = set(PARENT_OBJECT_RELATION_VALUES)

# A child can resolve to a parent only through one of these type-compatible
# containment pairs. Same-type nesting is allowed where the source genuinely
# names a subproject/program; every other cross-nature merge/parent is blocked.
# 2026-08-28 pilot fix: the matrix now actually CONTAINS the same-type pairs
# this comment always promised (the omission caused 9/67 first-pass failures
# in one real wave: power_project-in-power-complex, program-in-program,
# facility-in-facility are genuine source patterns).  Acquisition subjects
# stay parent-less; incompatible pairs are cleared+flagged at assembly and
# the registry independently re-checks this matrix before materializing any
# edge.
OBJECT_PARENT_TYPES = {
    "building": {"building", "phase", "facility", "campus", "data_center_project"},
    "phase": {"phase", "facility", "campus", "data_center_project", "program"},
    "facility": {"facility", "campus", "data_center_project", "program"},
    "campus": {"campus", "data_center_project", "program"},
    "data_center_project": {"data_center_project", "program"},
    "power_project": {"power_project", "program"},
    "acquisition_company": set(),
    "acquisition_business": set(),
    "acquisition_asset_portfolio": set(),
    "program": {"program"},
}

# Explicit DOCUMENTATION aid for old Stage-2/Fact exports. Every retired value
# is contextual and must never be silently rewritten: the suggested v2 axes
# identify what a re-extraction/human migration should inspect.
LEGACY_AMOUNT_BASIS_CROSSWALK = {
    "minimum_commitment": {
        "candidate_basis": "contract_value", "candidate_level": "unknown",
        "automatic": False},
    "annual_payment": {
        "candidate_basis": "contract_value", "candidate_level": "period_flow",
        "automatic": False},
    "unit_rate": {
        "candidate_basis": "contract_value", "candidate_level": "unit_rate",
        "automatic": False},
    "remaining_commitment": {
        "candidate_basis": "facility_commitment",
        "candidate_level": "remaining_balance", "automatic": False},
    "outstanding_debt": {
        "candidate_basis": "debt_principal",
        "candidate_level": "remaining_balance", "automatic": False},
}

# ---------------------------------------------------------------------------
# Source types + provenance preference (most auditable source wins on a tie).
# ---------------------------------------------------------------------------
SOURCE_RANK = {"sec_10-K": 4, "sec_10-Q": 4, "sec_20-F": 4,
               "sec_8-K": 3, "sec_8-K/A": 3, "sec_6-K": 3, "sec_6-K/A": 3,
               "transcript": 2, "reuters_news": 1, "dcd_news": 1}

# THE news predicate.  Every news-vs-filer decision (prompt header mode, the
# null-target default, --no-news) must go through is_news_source -- three
# hand-rolled predicates drifted apart when reuters_news joined the corpus
# (prompts.py knew it; classify.py/assemble.py did not), silently defaulting
# null-target Reuters observations to the chunk's hyperscaler firm tag and
# letting --no-news keep the majority news source.
NEWS_SOURCE_TYPE_PREFIXES = ("dcd_news",)
NEWS_SOURCE_TYPES = {"news", "reuters_news"}


def is_news_source(source_type) -> bool:
    """True iff the source_type is a news source (headline subject is never
    the default obligor; --no-news excludes it)."""
    st = str(source_type or "").strip().lower()
    return st.startswith(NEWS_SOURCE_TYPE_PREFIXES) or st in NEWS_SOURCE_TYPES
