#!/usr/bin/env python3
"""Stage `deduplicate_facts` -- observation -> Fact via tiered resolution.

Tier A  rule_exact        identical claim -> auto-link, no LLM
Tier B  rule_compatible   rounding/bound variants of one claim -> auto-link;
                          also a guidance restatement inside the 5% midpoint
                          tolerance (MRG-1)
Tier D  vetoes            currency / fiscal-period / target / measure-axis
                          conflicts -> cannot-link, recorded; also a guidance
                          revision at or beyond 5% (MRG-1), which MRG-2 then
                          chains by series_id.
                          UND-6 (2026-08-22): the fiscal-period veto fires on
                          PROVEN disjointness of the time layer's resolved
                          date envelopes, never on label-string mismatch
                          ('FY2025' vs '2025' vs '2025-07' describe
                          overlapping periods; only truly disjoint envelopes
                          are two different budgets).
                          UND-5 (2026-08-22): a separateness cue in the newer
                          quote no longer vetoes -- it routes the pair to the
                          judge with a hint.  The 2025 run audit showed the
                          regex fires on cues that separate a DIFFERENT
                          nearby amount (Meta's "In addition to lease
                          liabilities..." $58.14B lease), and on the pairs it
                          blocked the judge is the reliable authority.
Tier C  LLM pair          bounded judgment obs vs candidate Fact (<=5
                          candidates), abstain allowed; 'unsure' never merges.
                          MRG-3 demotes same-target/same-amount pairs that have
                          neither counterparty corroboration nor date proximity
                          into this tier rather than letting a rule fire.

Candidates come from shared fingerprint keys (multi-key blocking with a
coarse 1-significant-digit amount bucket for recall; the fine 2-digit bucket
only strengthens rules).  Observations are processed in ONE deterministic
(doc_date, raw_id) sequence -- the earliest disclosure seeds a Fact and later
re-tellings attach to it -- and every pair judgment is content-cached, so a
rerun replays the exact same sequence for free.  A merge that would cross an
explicit cannot-link is blocked and queued for review.  Fact ids are
content-addressed over sorted member raw_ids.

Scale note: the sequential gray-zone loop is correct but latency-bound at
full-sample size; parallelizing it means wave-parallel batches within
key-disjoint blocks, which the link-ledger design permits (documented, not
implemented).
"""
from __future__ import annotations

import os
from collections import defaultdict

from .common import (Manifest, SEPARATE_CUE_RE, SOURCE_RANK,
                     amount_interval, bucket_coarse, bucket_fine,
                     coerce_num, mw_value, parse_date, read_jsonl,
                     representative_amount, sha16, write_jsonl)
from ..progress import Heartbeat
from .judge import Judge, content_id

MAX_PAIR_CANDIDATES = 5

# MRG-1/MRG-2: the amount bases on which a firm RE-STATES a forward number for
# the same fiscal period -- i.e. the classes on which "guidance revision" is a
# meaningful notion at all.  Every other basis describes a concluded fact whose
# amount does not get revised in place.
GUIDANCE_BASES = {"capex_cash_flow", "authorized_amount"}

# PI decision 2026-08-06: a re-stated guidance range whose MIDPOINT moved less
# than this is a restatement (one fact); at or beyond it, a revision (two facts
# chained by series_id).  Cite as "revision threshold 5% (midpoint of stated
# range), PI decision 2026-08-06" in METHODS.
GUIDANCE_REVISION_TOL = 0.05

# MRG-3: below this separation a same-target/same-amount pair may still
# rule-merge without counterparty corroboration; beyond it, it must be judged.
DISTINCT_DEAL_GAP_DAYS = 90

# UND-6a: the time-claim granularities that denote a BUDGET / REPORTING
# PERIOD, and may therefore prove two amounts describe different periods.
# Deliberately excludes `day` (as-of instants mis-typed as periods -- see
# period_envelopes_of) and `unknown`.
PERIOD_GRANULARITIES = frozenset({
    "fiscal_year", "fiscal_quarter", "year", "quarter",
    "half_year", "multi_year", "month"})

FACT_PAIR_SYS = """You are the regv1 FACT REGISTRY JUDGE. Decide whether the NEW OBSERVATION asserts the SAME atomic capital fact as the EXISTING FACT (shown through its member observations). One atomic fact = one economic commitment, financing, purchase, investment, or capital expenditure of the displayed standardized magnitude into the displayed target.

SAME fact: repeated disclosure of one claim -- later lifecycle labels (announced -> planned -> under construction), rounding or bound variants ('up to $500bn', 'about $500bn', '$500bn'), a different source type, a point vs its own stated range, or the same commitment anchored once on the vehicle/SPV and once on its sponsor(s). Differing geographic or perimeter WORDING around one commitment ('US infrastructure' vs 'global data centers', 'AI data centers' vs 'data centers and infrastructure') is ALSO the same fact when the parties, magnitude and period agree -- answer same_fact; the scope disagreement is recorded downstream, never by splitting the fact.
DIFFERENT facts: a REVISED or updated amount ('increased from X to Y', 'now $27bn' after an earlier $29bn) is a DIFFERENT fact from the original -- revisions link at the deal layer, never here (EXCEPTION: a re-stated guidance range for the SAME fiscal period whose midpoint moved by less than ~5% is the same fact -- a rounding/restatement, not a revision); separate tranches; different explicitly named providers each making their own same-sized contribution; annual budgets for different fiscal periods; object_total vs component_amount/tranche/period_flow/cumulative_to_date/incremental_change/remaining_balance/unit_rate; parent-object total vs child-object amount; separately sized or explicitly additional commitments. Never repair a reversed target/counterparty here: upstream direction admission owns that error.
Read source slots literally: counterparty is not other_party is not parent. When structured fields and quotes differ -- including WHO makes the contribution -- rely on the quotes. If the evidence cannot establish sameness, answer different_fact or unsure -- never merge on similarity alone.

Reply JSON ONLY: {"decision": "same_fact" | "different_fact" | "unsure", "reason": "short evidence-based reason", "confidence": "high" | "medium" | "low"}"""


# MRG-3: prepended to the judge's user text for pairs the distinct-deal guard
# demoted.  Deterministic given the pair, so caching is unaffected.
DISTINCT_DEAL_HINT = (
    "NOTE: the rows share target and magnitude but were disclosed far apart "
    "with no common counterparty -- verify this is the SAME deal, not two "
    "different deals that coincidentally share a number.\n")

# UND-5: prepended when the newer quote carries a separateness cue.  The cue
# is EVIDENCE to weigh, not proof of separateness: it may declare THIS amount
# additional to the existing fact (two deals), or it may separate this amount
# from something else entirely in the same excerpt (one deal, re-told).
SEPARATE_CUE_HINT = (
    "NOTE: the newer quote contains separateness language ('in addition to' / "
    "'another' / 'separate' / 'a second').  Decide from the quotes whether it "
    "declares THIS amount additional to / separate from the EXISTING FACT "
    "shown, or whether it separates this amount from some OTHER item in the "
    "excerpt.  Only a cue that separates THESE two amounts makes them "
    "different facts.\n")


def _valid_pair(o) -> bool:
    return (isinstance(o, dict)
            and o.get("decision") in {"same_fact", "different_fact", "unsure"}
            and o.get("confidence") in {"high", "medium", "low"})


def _stated_mid(row: dict):
    """MRG-1's midpoint: the central value of the amount AS STATED.

    Deliberately NOT the midpoint of `amount_interval`.  That helper returns a
    PLAUSIBILITY interval, not the stated quantity -- `max` -> (0, hi) and
    `min` -> (lo, lo*4) -- so its midpoint is hi/2 and lo*2.5.  MRG-1's own
    wording ("equals the point for points") shows the intent is the stated
    central value; read literally off amount_interval it would score Meta's
    January "up to $65 billion" as 32.5B and veto it against the "$60-65
    billion" range that is the SAME guidance, and would inflate every "at
    least X" by 2.5x.  A genuine range therefore gets its true midpoint and
    every other bound reports the amount it actually states.
    """
    lo = coerce_num(row.get("amount_native_low"))
    hi = coerce_num(row.get("amount_native_high"))
    if (row.get("amount_bound_type") or "") == "range" and lo is not None and hi is not None:
        return (lo + hi) / 2.0
    return representative_amount(row)


def _gap_days(a, b):
    """Absolute separation of two doc_dates in days; None when either side is
    unparsable (MRG-3 treats that as 'no evidence of proximity')."""
    da, db = parse_date(a), parse_date(b)
    if da is None or db is None:
        return None
    return abs((da - db).days)


def _mantissa(v) -> str:
    """3-significant-digit mantissa, scale-free: 310.79e9 USD and 310.79e12
    KRW both give '3.108'.  UND-3 uses it to spot the same printed number
    living under two currencies (the Samsung W/$ class)."""
    return f"{float(v):.3e}".split("e")[0]


def _fy_labels(feat: dict) -> tuple:
    """The fiscal-YEAR labels carried by a feature.  `periods` mixes calendar
    year labels ('2025'), fiscal year labels ('FY2025'), and quarter labels
    ('2025-07'); a guidance series chains one fiscal YEAR's successive
    estimates, so the year-shaped labels key it.  PI ruling 2026-08-20:
    'FYyyyy' labels key a series too (previously the 4-char filter locked
    Microsoft-style fiscal ladders out of MRG-2 entirely); 'FY2025' and
    '2025' stay DISTINCT keys -- they are different periods for a non-January
    filer, and conflating them would chain two different ladders."""
    return tuple(sorted(str(p) for p in (feat.get("periods") or ())
                        if len(str(p)) == 4
                        or (len(str(p)) == 6 and str(p).startswith("FY"))))


# ---------------------------------------------------------------------------
# Per-observation features
# ---------------------------------------------------------------------------
def obs_features(row: dict, period_labels: set, period_envs: tuple = ()) -> dict:
    # entity id sets
    tset = tuple(sorted(i for i in (row.get("target_entity_ids") or []) if i))
    pset = tuple(sorted(i for i in (row.get("parent_entity_ids") or []) if i))
    cset = tuple(sorted(i for i in (row.get("counterparty_entity_ids") or []) if i))
    # amount, point/high/low
    rep = representative_amount(row)
    cur = (row.get("amount_currency") or "unknown").strip().upper() or "UNKNOWN"
    mwv = mw_value(row)
    return {
        "raw_id": row["raw_id"], "target_set": tset, "par_set": pset, "cp_set": cset,
        "object": row.get("object_id"),
        "parent_object": row.get("parent_object_id"),
        "object_type": row.get("object_type"),
        "basis": (row.get("amount_basis") or "unknown"),
        "aggregation_level": (row.get("amount_aggregation_level") or "unknown"),
        "instrument": (row.get("instrument_class") or "unsure"),
        "currency": cur,
        "rep": rep, "fine": bucket_fine(rep) if rep is not None else None,
        "coarse": bucket_coarse(rep) if rep is not None else None,
        # MRG-1 inputs: the amount's period scope (already on the row from
        # amounts.py via quality_gate, but invisible to dedup until now) and
        # the stated midpoint the 5% revision tolerance is measured on.
        "scope": row.get("amount_period_scope") or "unclear",
        "mid": _stated_mid(row),
        "interval": amount_interval(row), "bound": row.get("amount_bound_type") or "unclear",
        "mw": mwv, "mw_coarse": bucket_coarse(mwv) if mwv is not None else None,
        "periods": frozenset(period_labels),
        "period_envs": tuple(period_envs),
        "quote": str(row.get("verbatim_quote") or ""),
        "doc_date": str(row.get("doc_date") or "9999"),
        "point": coerce_num(row.get("amount_native_point")),
        "low": coerce_num(row.get("amount_native_low")),
        "high": coerce_num(row.get("amount_native_high")),
    }


def obs_keys(f: dict) -> list[tuple[int, tuple]]:
    """High-precision composite candidate keys in fixed precedence tiers.

    Every key contains at least two independent identity axes. A shared amount,
    entity, object, text/source type, or claim source alone can never nominate
    a candidate. Tiers are retrieval precedence, not merge decisions.
    """
    keys: list[tuple[int, tuple]] = []
    closure = sorted(set(f["target_set"]) | set(f.get("par_set") or ()))
    object_closure = tuple(x for x in (f.get("object"), f.get("parent_object")) if x)
    if f["rep"] is not None:
        buckets = sorted({f["coarse"],
                          bucket_coarse(f["rep"] * 1.06),
                          bucket_coarse(f["rep"] * 0.94)})
        if f["target_set"] and f["cp_set"] and f["fine"] is not None:
            keys.append((0, ("endpoints_fine", f["target_set"], f["cp_set"],
                             f["currency"], f["fine"])))
        if f["target_set"] and f.get("object") and f["fine"] is not None:
            keys.append((0, ("target_object_fine", f["target_set"], f["object"],
                             f["object_type"], f["currency"], f["fine"])))
        for b in buckets:
            for oid in object_closure:
                # Object ids are already typed. Adding the child's type here
                # would prevent two different child types from meeting on the
                # same parent object.
                keys.append((1, ("object_amount", oid,
                                 f["currency"], b)))
            if f["target_set"] and f["cp_set"]:
                keys.append((1, ("endpoints_instrument_amount", f["target_set"],
                                 f["cp_set"], f["instrument"], f["currency"], b)))
            for e in closure:
                keys.append((3, ("target_instrument_amount_level", e,
                                 f["instrument"], f["currency"], b,
                                 f["aggregation_level"])))
                # Lower-precedence recall backstop for an instrument caveat.
                # This composite key only nominates a cluster; pair rules or
                # the judge must still decide whether it is the same Fact.
                keys.append((3, ("target_basis_amount", e, f["basis"],
                                 f["currency"], b)))
        if f["target_set"] and f["cp_set"]:
            keys.append((2, ("endpoints_basis_level", f["target_set"], f["cp_set"],
                             f["basis"], f["aggregation_level"])))
        if object_closure:
            for e in closure:
                for oid in object_closure:
                    keys.append((2, ("target_object_relation", e, oid,
                                     f["instrument"], f["aggregation_level"])))
    if f["mw_coarse"] is not None:
        if f["target_set"] and f.get("object"):
            keys.append((0, ("capacity_target_object", f["target_set"],
                             f["object"], f["mw_coarse"])))
        if f["target_set"] and f["cp_set"]:
            keys.append((1, ("capacity_endpoints", f["target_set"],
                             f["cp_set"], f["mw_coarse"])))
        for e in closure:
            if object_closure:
                for oid in object_closure:
                    keys.append((2, ("capacity_target_object_closure", e, oid,
                                     f["mw_coarse"])))
    return keys


def period_labels_of(claims: list[dict]) -> set:
    labels = set()
    for c in claims:
        # One shared notion of "active claim" with the temporal profile: a
        # claim the time layer dropped (escalation: not_this_event) must not
        # feed the fiscal-period veto or the MRG-1 same-period gate either.
        if (c.get("claim_status") or "active") == "dropped":
            continue
        if c.get("role") != "period" or c.get("interval") != "period":
            continue
        if c.get("fiscal_year"):
            labels.add(str(c["fiscal_year"]))
        elif c.get("granularity") == "year" and c.get("date_start"):
            labels.add(str(c["date_start"])[:4])
        elif c.get("granularity") == "quarter" and c.get("date_start"):
            labels.add(str(c["date_start"])[:7])
    return labels


def period_envelopes_of(claims: list[dict]) -> tuple:
    """UND-6: the resolved (date_start, date_end) envelopes of an
    observation's active period claims.  The time layer already resolves
    fiscal labels onto real dates ('in fiscal 2025' for a June filer ->
    2024-07-01..2025-06-30), so envelope overlap is the honest period
    compatibility test; label strings are kept only for the MRG-1 gate and
    the MRG-2 series keys, where strictness is intended.  A claim without
    both dates contributes nothing -- an unresolvable period can never PROVE
    disjointness.

    Only BUDGET/REPORTING granularities count (UND-6a).  The first live run
    showed `day`-granularity period claims are as-of INSTANTS mis-typed as
    periods ('As of December 31, 2024', 'as at 31 December 2024'): their
    envelope is a single day, so it is trivially disjoint from any other and
    manufactured 354 spurious 'different fiscal period' vetoes -- on exactly
    the balance-sheet stock disclosures that most need to merge across
    quarters.  An instant is not evidence of a different budget period.
    `unknown` is excluded for the same reason: a veto needs positive
    evidence, not an unlabelled span."""
    envs = set()
    for c in claims:
        if (c.get("claim_status") or "active") == "dropped":
            continue
        if c.get("role") != "period" or c.get("interval") != "period":
            continue
        if c.get("granularity") not in PERIOD_GRANULARITIES:
            continue
        ds, de = c.get("date_start"), c.get("date_end")
        if ds and de and str(ds) <= str(de):
            envs.add((str(ds), str(de)))
    return tuple(sorted(envs))


def _envelopes_disjoint(a: tuple, b: tuple) -> bool:
    """True iff the ISO-date envelopes do not overlap at all."""
    return a[1] < b[0] or b[1] < a[0]


# ---------------------------------------------------------------------------
# Pair classification (tiers)
# ---------------------------------------------------------------------------
def _amounts_equal(a: dict, b: dict) -> bool:
    def eq(x, y):
        if x is None and y is None:
            return True
        if x is None or y is None:
            return False
        return abs(x - y) <= max(1.0, abs(x) * 1e-6)
    return eq(a["point"], b["point"]) and eq(a["low"], b["low"]) and eq(a["high"], b["high"])


_COVERING_BOUNDS = {frozenset(("exact", "approx")), frozenset(("exact", "max")),
                    frozenset(("exact", "min")), frozenset(("approx", "approx")),
                    frozenset(("approx", "max")), frozenset(("approx", "min"))}


def _intervals_overlap(a, b) -> bool:
    if a is None or b is None:
        return True
    return not (a[1] < b[0] or b[1] < a[0])


def classify_pair(o: dict, members: list[dict]) -> tuple[str, str]:
    """(tier, reason): tier in rule_exact | rule_compatible | veto | llm."""
    # -- vetoes first
    # currency conflict
    for m in members:
        if o["currency"] != "UNKNOWN" and m["currency"] != "UNKNOWN" \
                and o["currency"] != m["currency"]:
            return "veto", f"currency_conflict:{o['currency']}vs{m['currency']}"
    # fiscal period conflict -- UND-6: only PROVEN envelope disjointness
    # vetoes.  The old label-string intersection scored 'FY2025' vs '2025' vs
    # '2025-07' as conflicting periods and split one Microsoft $80B guidance
    # into four facts; the time layer's resolved envelopes overlap for all of
    # those, while FY2024-vs-FY2026 stays a genuine two-budget veto.
    fact_envs = set()
    for m in members:
        fact_envs.update(m.get("period_envs") or ())
    o_envs = o.get("period_envs") or ()
    if o_envs and fact_envs and all(_envelopes_disjoint(a, b)
                                    for a in o_envs for b in fact_envs):
        return "veto", "fiscal_period_conflict"
    # Target disjointness is normally fatal, but a shared typed object is a
    # strong reason to let the judge inspect a possible granularity/role issue.
    # Dedup never repairs direction; this route only prevents a premature veto.
    o_closure = set(o["target_set"]) | set(o.get("par_set") or ())
    if o["target_set"] and members:
        overlap = any((set(m["target_set"]) | set(m.get("par_set") or ())) & o_closure
                      for m in members if m["target_set"])
        if not overlap and all(m["target_set"] for m in members):
            same_object = bool(o.get("object")) and any(
                m.get("object") == o.get("object") for m in members)
            return (("llm", "target_conflict_same_object") if same_object
                    else ("veto", "disjoint_targets_and_parents"))
    # measurement-axis protection: a capacity-only claim and a money claim
    # are different facts; unit rates never merge with totals; materially
    # different capacities are different facts.
    def axis(x):
        return "money" if x["rep"] is not None else ("cap" if x["mw"] is not None else "other")
    if any(axis(o) != axis(m) for m in members):
        return "veto", "cross_measure_axis"
    if any((o.get("aggregation_level") == "unit_rate") !=
           (m.get("aggregation_level") == "unit_rate") for m in members):
        return "veto", "unit_rate_axis"
    # v2: total/component/tranche/cumulative/remainder/rate semantics define
    # separate atomic claims. Unknown is an abstention and does not veto.
    for m in members:
        a, b = o.get("aggregation_level"), m.get("aggregation_level")
        if a not in (None, "unknown") and b not in (None, "unknown") and a != b:
            return "veto", f"aggregation_level_conflict:{a}vs{b}"
    if axis(o) == "cap":
        for m in members:
            if m["mw"] and o["mw"] and abs(o["mw"] - m["mw"]) > 0.10 * max(o["mw"], m["mw"]):
                return "veto", "capacity_magnitude_conflict"

    # UND-5: a separateness cue in the newer quote routes the pair to the
    # judge instead of vetoing.  Placed AFTER the structural vetoes (those
    # stay free and certain) and BEFORE the rule tiers on purpose: a genuine
    # "another $500M facility" carries amounts identical to its sibling, and
    # Rule 1 would auto-merge the pair if the cue did not interrupt here.
    # The 2025 forensics: on split pairs this veto blocked, the merge was
    # correct ~93% of the time -- the cue usually separates a DIFFERENT
    # nearby item, which only the quotes can show.  The judge gets
    # SEPARATE_CUE_HINT and both quotes.
    newer = max([o] + members, key=lambda x: x["doc_date"])
    if newer is o and SEPARATE_CUE_RE.search(o["quote"]):
        return "llm", "separateness_cue_in_new_quote"

    # -- MRG-1 guidance-revision tolerance gate (PI ruling 2026-08-06: 5%
    # midpoint).  Placed AFTER the vetoes and BEFORE the rule tiers on purpose:
    # Rule 2 (bound_variant_same_bucket) otherwise auto-merges successive
    # guidance ranges -- 66-72 and 70-72 share a fine bucket, their intervals
    # overlap and both bounds are `range` -- so the pair-judge, whose prompt
    # correctly calls a revised amount a different fact, never gets asked.
    # This gate decides that class deterministically instead.
    if (o["basis"] in GUIDANCE_BASES and o.get("scope") == "per_year"
            and all(m["basis"] == o["basis"] for m in members)
            and all(m["target_set"] == o["target_set"] for m in members)
            and o["periods"] and any(o["periods"] & (m["periods"] or frozenset())
                                     for m in members)):
        mids = [m["mid"] for m in members if m["mid"]]
        if o["mid"] and mids:
            # nearest member: a fact accumulating one ladder of restatements
            # must not be judged against its oldest rung alone.
            delta = min(abs(o["mid"] - m) / m for m in mids)
            if delta < GUIDANCE_REVISION_TOL:
                return "rule_compatible", "guidance_restatement_lt5pct"
            return "veto", "guidance_revision_ge5pct"

    # -- rule tiers need a coherent existing fact (single target/basis/currency)
    basis_set = {m["basis"] for m in members}
    object_set = {m["object"] for m in members if m.get("object")}
    # current obs target fits any current fact
    same_target = all(m["target_set"] == o["target_set"] for m in members)
    # A missing object can repeat a named-object fact, but two different
    # primary objects never auto-merge merely through a shared ancestor.
    object_ok = (not o.get("object") or not object_set or o["object"] in object_set)
    cp_sets = {m["cp_set"] for m in members} # cp is counterparty
    cp_conflict = (o["cp_set"] and any(cp and cp != o["cp_set"] for cp in cp_sets))
    cp_compatible = not cp_conflict

    # -- MRG-3 distinct-deal guard: a shared target and a shared amount are NOT
    # by themselves enough to rule-merge.  The `t1` blocking key carries no
    # counterparty and no time window, and an EMPTY counterparty counts as
    # compatible above (cp_conflict fires only on a DIFFERENT non-empty cp), so
    # a cp-less retelling months later can rule-merge into an unrelated fact and
    # drag its first-disclosure date backwards.  Demand corroboration -- the
    # same counterparty on both sides -- or proximity in time; failing both, the
    # pair goes to the judge with an explicit hint (see run()).
    # Applies to the AMOUNT axis only (o["rep"] set): the guard's premise is
    # "same {target, amount} alone cannot rule-merge".  Capacity-only pairs
    # (rep None) keep their own rule, identical_capacity -- without this gate
    # the guard intercepted 104 capacity pairs under a reason naming an
    # amount they do not have.
    cp_corroborated = bool(o["cp_set"]) and any(o["cp_set"] == cp for cp in cp_sets if cp)
    if o["rep"] is not None and not cp_corroborated:
        gaps = [g for g in (_gap_days(o["doc_date"], m["doc_date"]) for m in members)
                if g is not None]
        # no parsable gap at all -> no evidence of proximity -> judge it
        if not gaps or min(gaps) > DISTINCT_DEAL_GAP_DAYS:
            return "llm", "same_amount_distant_no_cp"
    # Same guard on the CAPACITY axis (the money-only restriction left the MW
    # axis with NO time gate at all: a counterparty-less capacity retelling
    # months later could rule-merge into any same-target ~equal-MW cluster --
    # two equal phases of one campus being the realistic false merge).
    if o["rep"] is None and o["mw"] is not None and not cp_corroborated:
        gaps = [g for g in (_gap_days(o["doc_date"], m["doc_date"]) for m in members)
                if g is not None]
        if not gaps or min(gaps) > DISTINCT_DEAL_GAP_DAYS:
            return "llm", "same_capacity_distant_no_cp"

    # Rule 1: exactly the same
    # coherent fact (same target set, same basis, same currency, compatible project/counterparty) 
    # and native amounts identical to ~1e-6 → auto-link. Also identical capacities (≤1%).
    same_instrument = all(m.get("instrument") == o.get("instrument") for m in members)
    level_ok = all(m.get("aggregation_level") in (o.get("aggregation_level"), "unknown")
                   or o.get("aggregation_level") == "unknown" for m in members)
    if same_target and object_ok and cp_compatible and same_instrument and level_ok \
            and {o["basis"]} == basis_set \
            and all(m["currency"] == o["currency"] for m in members):
        if all(_amounts_equal(o, m) for m in members):
            return "rule_exact", "identical_claim"
        if o["rep"] is not None:
            fines = {m["fine"] for m in members}
            bounds = {m["bound"] for m in members}
            # Rule 2: compatible -- absorbs "up to $500bn" vs "about $500bn"
            # vs "$500bn" (same fine bucket, overlapping intervals, covering
            # bound types).
            if fines == {o["fine"]} and all(_intervals_overlap(o["interval"], m["interval"])
                                            for m in members):
                pairs_ok = all(frozenset((o["bound"], b)) in _COVERING_BOUNDS
                               or o["bound"] == b for b in bounds)
                if pairs_ok:
                    return "rule_compatible", "bound_variant_same_bucket"
        if o["mw"] is not None and o["rep"] is None \
                and all(m["mw"] is not None and m["rep"] is None for m in members):
            if all(abs(o["mw"] - m["mw"]) <= 0.01 * max(o["mw"], m["mw"]) for m in members):
                return "rule_exact", "identical_capacity"
    return "llm", "gray_zone"


# ---------------------------------------------------------------------------
# Fact store (incremental clusters + key index + cannot-links)
# ---------------------------------------------------------------------------
class FactStore:
    def __init__(self):
        self.facts: dict[int, dict] = {}
        self.key_index: dict[tuple, set] = defaultdict(set)
        self.cannot: set[frozenset] = set()      # raw_id pairs judged different
        # raw_id pairs the judge answered `unsure` on: never a merge veto for
        # the observation itself, but a BRIDGE across an unsure pair is
        # blocked -- the sole bridge guard used to be cannot-links, which are
        # absent exactly on the pairs already established as ambiguous.
        self.unsure: set[frozenset] = set()
        self.next = 1

    def create(self, feat: dict) -> int:
        fid = self.next
        self.next += 1
        self.facts[fid] = {"members": [feat]}
        for tier, key in obs_keys(feat):
            self.key_index[(tier, key)].add(fid)
        return fid

    def attach(self, fid: int, feat: dict) -> None:
        self.facts[fid]["members"].append(feat)
        for tier, key in obs_keys(feat):
            self.key_index[(tier, key)].add(fid)

    def candidates(self, feat: dict) -> tuple[list[tuple[int, dict]], int]:
        """Return Top-5 clusters by lexicographic composite-key precedence.

        The second return value is the uncapped candidate-pool size for recall
        telemetry. Weak evidence cannot add its way above a stronger tier.
        """
        matches: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
        for tier, key in obs_keys(feat):
            for fid in self.key_index.get((tier, key), set()):
                if fid in self.facts:
                    matches[fid].append((tier, key))

        def rank_item(fid: int):
            ms = matches[fid]
            best = min(t for t, _k in ms)
            members = self.facts[fid]["members"]
            endpoint = any(m["target_set"] == feat["target_set"]
                           and m["cp_set"] == feat["cp_set"]
                           and bool(feat["cp_set"]) for m in members)
            object_same = bool(feat.get("object")) and any(
                m.get("object") == feat.get("object") for m in members)
            fine = feat.get("fine") is not None and any(
                m.get("fine") == feat.get("fine") for m in members)
            basis = any(m.get("basis") == feat.get("basis") for m in members)
            instrument = any(m.get("instrument") == feat.get("instrument") for m in members)
            level = any(m.get("aggregation_level") == feat.get("aggregation_level")
                        for m in members)
            gaps = [g for g in (_gap_days(feat.get("doc_date"), m.get("doc_date"))
                                for m in members) if g is not None]
            min_gap = min(gaps) if gaps else 10 ** 9
            first_member = min(m["raw_id"] for m in members)
            return (best, -sum(1 for t, _k in ms if t == best), -len(ms),
                    -int(endpoint), -int(object_same), -int(fine), -int(basis),
                    -int(instrument), -int(level), min_gap, first_member, fid)

        ordered = sorted(matches, key=rank_item)
        out = []
        for rank, fid in enumerate(ordered[:MAX_PAIR_CANDIDATES], 1):
            ms = matches[fid]
            out.append((fid, {
                "candidate_rank": rank,
                "best_key_tier": min(t for t, _k in ms),
                "matched_key_count": len(ms),
                "matched_key_names": sorted({k[0] for _t, k in ms}),
            }))
        return out, len(matches)

    def cannot_between(self, fa: int, fb: int) -> bool:
        ra = [m["raw_id"] for m in self.facts[fa]["members"]]
        rb = [m["raw_id"] for m in self.facts[fb]["members"]]
        return any(frozenset((x, y)) in self.cannot for x in ra for y in rb)

    def unsure_between(self, fa: int, fb: int) -> bool:
        ra = [m["raw_id"] for m in self.facts[fa]["members"]]
        rb = [m["raw_id"] for m in self.facts[fb]["members"]]
        return any(frozenset((x, y)) in self.unsure for x in ra for y in rb)

    def merge(self, keep: int, absorb: int) -> None:
        for feat in self.facts[absorb]["members"]:
            self.facts[keep]["members"].append(feat)
        for fids in self.key_index.values():
            if absorb in fids:
                fids.discard(absorb)
                fids.add(keep)
        del self.facts[absorb]


def _card(row: dict) -> str:
    return "\n".join([
        f"raw_id={row['raw_id']} doc_date={row.get('doc_date')} source={row.get('source_type')}",
        f"target={';'.join([c for c in (row.get('target_canonicals') or []) if c]) or row.get('target_firm')}"
        f" | counterparty={row.get('counterparty')} | parent={row.get('target_parent')}",
        f"object={row.get('object_canonical') or row.get('object_name')}"
        f" | object_type={row.get('object_type')}"
        f" | instrument={row.get('instrument_class')}"
        f" | debt_subtype={row.get('debt_instrument_subtype')}"
        f" | basis={row.get('amount_basis')}"
        f" | aggregation_level={row.get('amount_aggregation_level')}",
        f"amount_text={row.get('amount_text')!r} parsed=[point={row.get('amount_native_point')},"
        f"low={row.get('amount_native_low')},high={row.get('amount_native_high')},"
        f"{row.get('amount_currency')},{row.get('amount_bound_type')}]"
        f" quantity={row.get('quantity_value')} {row.get('quantity_unit') or ''}",
        f"deal_status={row.get('deal_status')} | commitment={row.get('commitment_status')}",
        f"summary: {str(row.get('summary') or '')[:200]}",
        f"quote: {str(row.get('verbatim_quote') or '')[:300]}",
    ])


def _member_cards(members: list[dict], obs_by_id: dict, cap: int = 3) -> list[str]:
    ordered = sorted(members, key=lambda m: (m["doc_date"], m["raw_id"]))
    picks = [ordered[0]]
    if len(ordered) > 1 and ordered[-1] is not ordered[0]:
        picks.append(ordered[-1])
    for m in ordered:
        if len(picks) >= cap:
            break
        if m not in picks:
            picks.append(m)
    return [_card(obs_by_id[m["raw_id"]]) for m in picks[:cap]]

def _bridge_conflict(store: "FactStore", keep: int, absorb: int):
    """Cluster-vs-cluster compatibility check before a bridge merge.

    A bridge merges two EXISTING clusters because one new observation is
    auto-compatible with each -- but the clusters themselves were never
    compared (a null-object / counterparty-less observation is compatible
    with anything, so it can bridge cluster A (object X) and cluster B
    (object Y) that would have vetoed each other directly).  Returns a
    conflict reason, or None when the bridge is safe:
      * any member of `absorb` that classify_pair VETOES against `keep`;
      * two clusters carrying disjoint non-null primary objects;
      * two clusters carrying disjoint non-empty counterparty sets."""
    km = store.facts[keep]["members"]
    am = store.facts[absorb]["members"]
    for mf in am:
        tier, reason = classify_pair(mf, km)
        if tier == "veto":
            return f"veto:{reason}"
    kobj = {m.get("object") for m in km if m.get("object")}
    aobj = {m.get("object") for m in am if m.get("object")}
    if kobj and aobj and not (kobj & aobj):
        return "distinct_objects"
    kcp = {m["cp_set"] for m in km if m["cp_set"]}
    acp = {m["cp_set"] for m in am if m["cp_set"]}
    if kcp and acp and not (kcp & acp):
        return "distinct_counterparties"
    return None


# rank the obs in the fact and select the representative obs as canonical
def _representative(raw_ids: list[str], obs_by_id: dict) -> str:
    def score(rid):
        row = obs_by_id[rid]
        flags = row.get("intake_flags") or {}
        return (SOURCE_RANK.get(row.get("source_type"), 1) * 100,
                -len(flags.get("audit_bad", [])) * 20,
                -len(flags.get("audit_unsure", [])) * 3,
                -int(bool(row.get("amount_needs_review"))) * 5,
                -int(bool(row.get("time_needs_review"))) * 5,
                min(len(str(row.get("verbatim_quote") or "")), 1200) // 100,
                rid)
    return max(raw_ids, key=score)


# ---------------------------------------------------------------------------
def run(outdir: str, judge: Judge) -> dict:
    man = Manifest(outdir, "deduplicate_facts")
    obs_path = os.path.join(outdir, "observations_final.jsonl")
    claims_path = os.path.join(outdir, "obs_time_claims.jsonl")
    man.add_input(obs_path)
    man.add_input(claims_path)
    rows = read_jsonl(obs_path)
    obs_by_id = {r["raw_id"]: r for r in rows}
    claims_by = defaultdict(list)
    for c in read_jsonl(claims_path):
        claims_by[c["raw_id"]].append(c)

    eligible = [r for r in rows if r["observation_status"] != "quarantined"]
    eligible.sort(key=lambda r: (str(r.get("doc_date") or "9999"), r["raw_id"]))

    store = FactStore()
    links: list[dict] = []
    reviews: list[dict] = []
    tier_counts = defaultdict(int)
    n_llm_same = n_llm_diff = n_llm_unsure = 0
    candidate_pool_total = candidate_returned_total = candidate_pool_max = 0
    retrieval_tier_counts = defaultdict(int)

    def link_row(feat, member_ids, tier, decision, source, reason,
                 escalation=None, retrieval=None):
        """`reason` justifies the VERDICT (a rule name, or the judge's own
        words).  `escalation_reason` records why the pair reached the judge at
        all -- on the llm tier the judge's reason overwrites the classifier's,
        so without this an MRG-3 demotion leaves no trace in the ledger and
        cannot be counted (VAL-1 / the over-merge rescan both need it)."""
        links.append({"raw_id": feat["raw_id"], "fact_members": member_ids,
                      "tier": tier, "decision": decision,
                      "decision_source": source, "reason": str(reason)[:300],
                      "escalation_reason": escalation,
                      "candidate_retrieval": retrieval})

    hb = Heartbeat("fact_dedup", total=len(eligible))
    for _hb_done, r in enumerate(eligible):
        hb.beat(_hb_done, lambda: (
            f"facts {len(store.facts)}"
            f" | llm same/diff/unsure {n_llm_same}/{n_llm_diff}/{n_llm_unsure}"))
        feat = obs_features(r, period_labels_of(claims_by[r["raw_id"]]),
                            period_envelopes_of(claims_by[r["raw_id"]]))
        auto_hits, llm_cands = [], []
        candidates, pool_size = store.candidates(feat)
        candidate_pool_total += pool_size
        candidate_pool_max = max(candidate_pool_max, pool_size)
        candidate_returned_total += len(candidates)
        for fid, retrieval in candidates:
            retrieval_tier_counts[retrieval["best_key_tier"]] += 1
            members = store.facts[fid]["members"]
            tier, reason = classify_pair(feat, members)
            tier_counts[tier] += 1
            mem_ids = sorted(m["raw_id"] for m in members)
            if tier in ("rule_exact", "rule_compatible"):
                auto_hits.append((fid, tier, reason, retrieval))
            elif tier == "veto":
                for m in members:
                    store.cannot.add(frozenset((feat["raw_id"], m["raw_id"])))
                link_row(feat, mem_ids, tier, "different", f"rule:{reason}",
                         reason, retrieval=retrieval)
                # UND-3 (A4): the currency veto is correct, but when the two
                # sides print the SAME number under different currencies
                # (W310.79T vs $310.79T) that is one claim mis-parsed, not two
                # facts -- surface it as a review row instead of staying silent.
                if reason.startswith("currency_conflict"):
                    same_mantissa = any(_mantissa(feat["rep"]) == _mantissa(m["rep"])
                                        for m in members if m["rep"] and feat["rep"])
                    if same_mantissa:
                        reviews.append({"stage": "fact",
                                        "reason": "currency_conflict_mantissa_match",
                                        "raw_id": feat["raw_id"],
                                        "members": mem_ids,
                                        "detail": reason})
            # tier dispatch into the LLM bucket -- the reason travels with the
            # candidate so MRG-3 can hint the judge on the pair it flagged.
            else:
                llm_cands.append((fid, reason, retrieval))

        if auto_hits:
            keep, tier0, reason0, retrieval0 = auto_hits[0]
            link_row(feat, sorted(m["raw_id"] for m in store.facts[keep]["members"]),
                     tier0, "same", f"rule:{tier0}", reason0,
                     retrieval=retrieval0)
            store.attach(keep, feat)
            for fid, tier, reason, retrieval in auto_hits[1:]:
                if fid not in store.facts:
                    continue
                if store.cannot_between(keep, fid):
                    reviews.append({"stage": "fact",
                                    "reason": "merge_blocked_by_cannot_link",
                                    "raw_ids": [feat["raw_id"]],
                                    "detail": {"kept_members":
                                               sorted(m["raw_id"] for m in store.facts[keep]["members"]),
                                               "blocked_members":
                                               sorted(m["raw_id"] for m in store.facts[fid]["members"])}})
                    continue
                # A bridge is a cluster-vs-cluster merge the pair rules never
                # examined: verify the two clusters directly, and never
                # bridge across a pair the judge already found ambiguous.
                conflict = (("unsure_pair" if store.unsure_between(keep, fid)
                             else None) or _bridge_conflict(store, keep, fid))
                if conflict:
                    reviews.append({"stage": "fact",
                                    "reason": "bridge_blocked_by_conflict",
                                    "raw_ids": [feat["raw_id"]],
                                    "detail": {"conflict": conflict,
                                               "kept_members":
                                               sorted(m["raw_id"] for m in store.facts[keep]["members"]),
                                               "blocked_members":
                                               sorted(m["raw_id"] for m in store.facts[fid]["members"])}})
                    continue
                store.merge(keep, fid)
                link_row(feat, sorted(m["raw_id"] for m in store.facts[keep]["members"]),
                          tier, "same_bridge_merge", f"rule:{tier}", reason,
                          retrieval=retrieval)
            continue

        placed = False
        attached_fid = None

        # Retrieval already applied the global Top-5 cap using lexicographic
        # composite-key precedence. Preserve that order here; a secondary
        # additive score must not reorder a weak candidate above a strong key.
        # Every candidate is judged even after a same_fact attach: the old
        # first-match break left a second same-Fact cluster permanently split
        # (judge-tier verdicts never bridged), which is exactly the head-
        # splitting under-merge the 2025 audit measured.
        ranked = sorted(llm_cands,
                        key=lambda fr: fr[2].get("candidate_rank", 10 ** 9))
        for fid, cand_reason, retrieval in ranked:
            if fid not in store.facts or fid == attached_fid:
                continue
            members = store.facts[fid]["members"]
            mem_ids = sorted(m["raw_id"] for m in members)
            # deterministic function of the pair, so the content cache stays sane
            hint = (DISTINCT_DEAL_HINT
                    if cand_reason in ("same_amount_distant_no_cp",
                                       "same_capacity_distant_no_cp") else
                    SEPARATE_CUE_HINT
                    if cand_reason == "separateness_cue_in_new_quote" else "")
            user = hint + "\n\n".join(
                ["NEW OBSERVATION:", _card(obs_by_id[feat["raw_id"]]),
                 "EXISTING FACT (member observations shown):"] +
                _member_cards(members, obs_by_id))
            # The item id folds in a digest of the RENDERED card text: member
            # ids alone are stable under upstream election/resolution changes,
            # so a member-stable pair could replay a verdict rendered on
            # different card content (fact_relations and the time layer both
            # already digest their rendered prompts for this reason).
            item = content_id("factpair", feat["raw_id"], *mem_ids, sha16(user))
            obj = judge.cached_json("r5_fact_pair", item, FACT_PAIR_SYS, user,
                                    validate=_valid_pair)
            if obj is None or obj["decision"] == "unsure":
                n_llm_unsure += 1
                for m in members:
                    store.unsure.add(frozenset((feat["raw_id"], m["raw_id"])))
                link_row(feat, mem_ids, "llm", "unsure", "llm",
                          (obj or {}).get("reason", "quarantine"), cand_reason,
                          retrieval)
                reviews.append({"stage": "fact", "reason": "llm_unsure_pair",
                                "raw_ids": [feat["raw_id"]] + mem_ids,
                                "detail": {"reason": (obj or {}).get("reason")}})
                continue
            if obj["decision"] == "same_fact":
                n_llm_same += 1
                if not placed:
                    link_row(feat, mem_ids, "llm", "same", "llm",
                             obj.get("reason", ""), cand_reason, retrieval)
                    store.attach(fid, feat)
                    placed = True
                    attached_fid = fid
                    continue
                # A SECOND same-Fact verdict: merge the clusters, subject to
                # the same guards as a rule-tier bridge.
                conflict = (("cannot_link" if store.cannot_between(attached_fid, fid)
                             else None)
                            or ("unsure_pair" if store.unsure_between(attached_fid, fid)
                                else None)
                            or _bridge_conflict(store, attached_fid, fid))
                if conflict:
                    reviews.append({"stage": "fact",
                                    "reason": "llm_bridge_blocked_by_conflict",
                                    "raw_ids": [feat["raw_id"]] + mem_ids,
                                    "detail": {"conflict": conflict}})
                    continue
                store.merge(attached_fid, fid)
                link_row(feat,
                         sorted(m["raw_id"] for m in store.facts[attached_fid]["members"]),
                         "llm", "same_bridge_merge", "llm",
                         obj.get("reason", ""), cand_reason, retrieval)
                continue
            n_llm_diff += 1
            for m in members:
                store.cannot.add(frozenset((feat["raw_id"], m["raw_id"])))
            link_row(feat, mem_ids, "llm", "different", "llm",
                     obj.get("reason", ""), cand_reason, retrieval)
        if not placed:
            fid = store.create(feat)
            link_row(feat, [feat["raw_id"]], "singleton", "new",
                     "rule:new_fact", "no compatible candidate")

    # ---- materialize (content-addressed ids, deterministic order)
    fact_rows, member_rows = [], []
    clusters = sorted(store.facts.values(),
                      key=lambda f: (min(m["doc_date"] for m in f["members"]),
                                     min(m["raw_id"] for m in f["members"])))
    cluster_of = {}
    for f in clusters:
        raw_ids = sorted(m["raw_id"] for m in f["members"])
        fact_id = "RF-" + sha16(*raw_ids)
        rep = _representative(raw_ids, obs_by_id)
        collective = any(obs_by_id[r].get("is_collective_target") for r in raw_ids)
        # UND-2 (T3): members may describe one commitment with different
        # geographic/perimeter wording ("US infrastructure" vs "global data
        # centers").  Per the PI ruling the fact still merges (UND-4 gives the
        # judge permission), but the disagreement is SURFACED, never silently
        # resolved: analysis decides which scope to report, and every member
        # quote is preserved by construction.
        geo = {str(obs_by_id[r].get("geo_scope") or "") for r in raw_ids}
        geo -= {"", "None", "unknown", "unclear", "null", "na"}
        fact_rows.append({"fact_id": fact_id, "raw_ids": raw_ids,
                          "canonical_observation_id": rep,
                          "is_collective_target": collective,
                          "scope_conflict": len(geo) > 1,
                          "n_observations": len(raw_ids),
                          # MRG-2, filled below; declared here so the column
                          # exists on every row.
                          "series_id": None})
        cluster_of[fact_id] = f
        for rid in raw_ids:
            member_rows.append({"fact_id": fact_id, "raw_id": rid})

    # ---- MRG-2: chain the guidance revisions that MRG-1 deliberately kept
    # apart.  Amount anchors fact identity (correct), so Meta's 62 -> 68 -> 69
    # ladder is separate facts; the SERIES is what makes the ladder visible.
    # Keyed on the canonical observation's (target, basis, fiscal year) --
    # NEVER derived from fact_ids, which are content-addressed over membership
    # (ground rule 5).
    series_members = defaultdict(list)
    for fr in fact_rows:
        f = cluster_of[fr["fact_id"]]
        feat = next((m for m in f["members"]
                     if m["raw_id"] == fr["canonical_observation_id"]), None)
        if feat is None:
            continue
        fy = _fy_labels(feat)
        if not (feat["basis"] in GUIDANCE_BASES and feat.get("scope") == "per_year" and fy):
            continue
        sid = "GS-" + sha16(*feat["target_set"], feat["basis"], *fy)
        fr["series_id"] = sid
        series_members[sid].append(
            {"fact_id": fr["fact_id"],
             "first_disclosed_at": min(m["doc_date"] for m in f["members"]),
             "mid": feat["mid"], "n_observations": fr["n_observations"],
             "_target": list(feat["target_set"]), "_basis": feat["basis"], "_fy": list(fy)})

    series_rows = []
    for sid in sorted(series_members):
        items = sorted(series_members[sid],
                       key=lambda d: (d["first_disclosed_at"], d["fact_id"]))
        head = items[0]
        series_rows.append({
            "series_id": sid,
            "target_entity_ids": head["_target"],
            "amount_basis": head["_basis"],
            "fiscal_year_labels": head["_fy"],
            "n_facts": len(items),
            "facts": [{k: v for k, v in d.items() if not k.startswith("_")}
                      for d in items],
        })

    paths = {"facts": os.path.join(outdir, "facts.jsonl"),
             "members": os.path.join(outdir, "fact_members.jsonl"),
             "links": os.path.join(outdir, "fact_links.jsonl"),
             "reviews": os.path.join(outdir, "fact_reviews.jsonl"),
             "series": os.path.join(outdir, "guidance_series.jsonl")}
    write_jsonl(fact_rows, paths["facts"])
    write_jsonl(member_rows, paths["members"])
    write_jsonl(links, paths["links"])
    write_jsonl(reviews, paths["reviews"])
    write_jsonl(series_rows, paths["series"])
    for p in paths.values():
        man.add_output(p)

    stats = {"observations_eligible": len(eligible),
             "facts": len(fact_rows),
             "guidance_series": len(series_rows),
             "facts_in_a_series": sum(1 for f in fact_rows if f["series_id"]),
             "scope_conflict_facts": sum(1 for f in fact_rows if f["scope_conflict"]),
             "currency_conflict_reviews": sum(
                 1 for r in reviews if r["reason"] == "currency_conflict_mantissa_match"),
              "multi_observation_facts": sum(1 for f in fact_rows if f["n_observations"] > 1),
              "candidate_top_k": MAX_PAIR_CANDIDATES,
              "candidate_pool_total": candidate_pool_total,
              "candidate_pool_max": candidate_pool_max,
              "candidate_returned_total": candidate_returned_total,
              "candidate_retrieval_tier_counts": dict(retrieval_tier_counts),
              "tier_counts": dict(tier_counts),
             "llm_same": n_llm_same, "llm_different": n_llm_diff,
             "llm_unsure": n_llm_unsure, "reviews": len(reviews)}
    man.done(stats, judge.report())
    return stats
