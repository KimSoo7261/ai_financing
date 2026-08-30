#!/usr/bin/env python3
"""Stage `build_fact_state` -- deterministic per-Fact economic projection.

No LLM.  A Fact is identity-only (an id + member observation ids); this stage
projects its economic face:

  * fact_state.jsonl        one row per Fact -- union-with-provenance entity /
                            object / basis / currency sets over the members,
                            ELECTED canonical values (CAN-1), and the full
                            dated amount evidence list.  Disagreement between
                            members is PRESERVED (the amounts list, the
                            contested flags), never averaged away.
  * fact_participants.jsonl one row per Fact x member x entity x source slot
                            (the edge list the network view filters by date).

CAN-1 (2026-08-22, PI decision): canonical values are ELECTED PER FIELD over
the members, replacing the single-representative dictatorship (the old rule
let one source-ranked row win every field; on the 2025 run 62% of
multi-member facts canonicalized a row carrying an audit caveat while a clean
member sat in the same cluster, and 136 facts took a vaguer bound than a
member offered).  The design splits fields by kind:

  ELECTED (one true value per fact, best estimate wins):
    amount            -- members are grouped into VALUE groups (2-significant-
                         digit + currency); the winning group is chosen by
                         source rank first (an SEC filing's figure beats any
                         count of wire re-prints -- news articles copy each
                         other, so document count is a correlated vote), then
                         bound precision (exact > approx > range > max/min),
                         then LATEST doc_date (a restated guidance range
                         supersedes its opener), then document support; inside
                         the winning group the most precise representation
                         wins ("$5 billion" exact over "up to $5.0 billion").
    enum fields       -- amount_basis, instrument_class, economic_action,
                         amount_period_scope, ai_infra_scope, geo_scope,
                         conditionality: majority vote, one vote per DOCUMENT
                         (a re-printed wire story is one telling, not many),
                         uninformative values (unknown/unsure/unclear) never
                         vote, a member whose audit graded that field bad is
                         disqualified from that field only; ties break to the
                         audit-clean carrier, then source rank, then LATEST
                         doc_date (an earliest-doc tiebreak would report the
                         stalest print of a revised number).
  SERIAL (the value tracks the telling, not the fact -- never canonical):
    deal_status       -- what each excerpt DID (announce / re-report /
                         update); shipped as deal_status_series.
    commitment_status -- the real-world stage PROGRESSES, so a majority vote
                         freezes it at the most-repeated early rung; shipped
                         as commitment_status_series plus a
                         commitment_status_latest convenience (latest
                         informative telling), explicitly derived-from-series.
  UNION (identity evidence, kept as before):
    entity/object ids, bases, currencies, instrument_classes, amounts[].

`canonical_observation_id` remains the PROVENANCE anchor (quote, summary,
citation) chosen by fact_dedup._representative; it no longer dictates values.
Every elected field carries vote counts in election_meta and a contested
flag, so member disagreement is a filterable quality signal.

The Fact's timeline is NOT built here -- build_temporal_profile owns that.
"""
from __future__ import annotations

import math
import os

from .common import Manifest, SOURCE_RANK, mw_value, read_jsonl, write_jsonl

# values that never vote: the model abstained, they carry no information
_UNINFORMATIVE = {None, "", "unknown", "unsure", "unclear", "na", "none", "null"}

# audit check that disqualifies a member from voting on ONE field (the
# auditor graded exactly that field bad on exactly that observation)
_FIELD_AUDIT = {"amount_basis": "amount_basis_ok",
                "amount_aggregation_level": "amount_basis_ok",
                "instrument_class": "instrument_ok",
                "debt_instrument_subtype": "instrument_ok",
                "object_type": "object_ok",
                "ai_infra_scope": "ai_scope_ok",
                "claim_source": "in_scope"}

# Amount flags that mark a fallback FAILURE (the LLM lane failed and the
# distrusted deterministic values were kept).  A member carrying one is
# demoted -- never disqualified -- in the amount election: a clean member
# always outranks it, but a single-member fact still reports its value.
_AMOUNT_FALLBACK_FLAGS = frozenset({
    "llm_all_null", "llm_quarantine_det_fallback",
    "llm_not_reconstructible_det_fallback", "rule_not_reconstructible"})


def _amount_clean(member: dict) -> bool:
    """Clean amount evidence: canonical-eligible status, no amount-audit bad,
    no fallback-failure flag.  The amount election prefers clean members at
    every ranking step (CAN-1's founding failure was exactly a caveated row
    out-electing a clean one on source rank alone)."""
    if member.get("observation_status") != "canonical_eligible":
        return False
    flags = member.get("intake_flags") or {}
    if "amount_ok" in (flags.get("audit_bad") or []):
        return False
    return not (set(member.get("amount_flags") or []) & _AMOUNT_FALLBACK_FLAGS)

# (output column, observation field) pairs elected by majority vote
_ELECTED_ENUMS = [
    ("canonical_basis", "amount_basis"),
    ("canonical_aggregation_level", "amount_aggregation_level"),
    ("canonical_instrument", "instrument_class"),
    ("canonical_debt_subtype", "debt_instrument_subtype"),
    ("canonical_action", "economic_action"),
    ("canonical_claim_source", "claim_source"),
    ("canonical_object_type", "object_type"),
    ("canonical_period_scope", "amount_period_scope"),
    ("canonical_ai_scope", "ai_infra_scope"),
    ("canonical_geo_scope", "geo_scope"),
    ("canonical_conditionality", "conditionality"),
]

_BOUND_PRECISION = {"exact": 4, "approx": 3, "range": 2, "max": 1, "min": 1}


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _audit_bad(member: dict, field: str) -> bool:
    check = _FIELD_AUDIT.get(field)
    if not check:
        return False
    flags = member.get("intake_flags") or {}
    return check in (flags.get("audit_bad") or [])


def _doc_key(member: dict) -> str:
    return str(member.get("doc_id") or member["raw_id"])


def _src(member: dict) -> int:
    return SOURCE_RANK.get(member.get("source_type"), 1)


def _date(member: dict) -> str:
    return str(member.get("doc_date") or "")


def _elect_enum(members: list[dict], field: str):
    """Majority vote, one vote per document.  Returns (value|None, meta).

    Two tiers.  The election proper counts only INFORMATIVE values from
    audit-eligible members.  When that leaves no voter at all, a fallback
    vote over every non-null value returns the modal raw value instead of
    None (CAN-1a): `unknown` / `unsure` / `unclear` are legitimate members
    of these enums, and a Fact whose every telling said `unknown` should
    report `unknown`, not a null that no downstream filter expects.  The
    tier that fired is recorded in `election_meta.fallback`."""
    def vote(pool):
        docs: dict = {}
        for m in pool:
            docs.setdefault(m[field], set()).add(_doc_key(m))
        if not docs:
            return None, docs

        def carrier_key(value):
            carriers = [m for m in pool if m[field] == value]
            clean = any(not (m.get("intake_flags") or {}).get("audit_bad")
                        for m in carriers)
            return (len(docs[value]), clean,
                    max(_src(m) for m in carriers),
                    max(_date(m) for m in carriers),
                    str(value))

        return max(docs, key=carrier_key), docs

    informative = [m for m in members
                   if m.get(field) not in _UNINFORMATIVE
                   and not _audit_bad(m, field)]
    winner, docs_for = vote(informative)
    fallback = False
    if winner is None:
        fallback = True
        winner, docs_for = vote([m for m in members
                                 if m.get(field) not in (None, "", "null")])
    if winner is None:
        return None, {"docs": 0, "values": 0, "contested": False,
                      "fallback": False}
    return winner, {"docs": len(docs_for[winner]),
                    "values": len(docs_for),
                    # a fallback election is not a real disagreement between
                    # informative tellings -- do not flag it as contested
                    "contested": (not fallback) and len(docs_for) > 1,
                    "fallback": fallback}


def _rep_amount(member: dict):
    for k in ("amount_native_point", "amount_native_mid",
              "amount_native_high", "amount_native_low"):
        v = member.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _round2sig(v: float) -> float:
    if not v:
        return 0.0
    exp = math.floor(math.log10(abs(v)))
    return round(v, -int(exp) + 1)


def _amount_dict(member: dict) -> dict:
    return {"point": member.get("amount_native_point"),
            "low": member.get("amount_native_low"),
            "high": member.get("amount_native_high"),
            "currency": member.get("amount_currency"),
            "bound_type": member.get("amount_bound_type"),
            "usd_reported": member.get("amount_usd_reported"),          # AMT-2
            "usd_reported_source": member.get("usd_reported_source"),   # AMT-2
            "amount_method": member.get("amount_method"),
            "amount_flags": member.get("amount_flags") or [],
            "observation_status": member.get("observation_status"),
            "blocking_reasons": member.get("blocking_reasons") or []}


def _elect_amount(members: list[dict], canon: dict):
    """Value first (2-sig + currency groups, doc-weighted), then precision of
    representation inside the winning value.  Returns
    (amount_dict, provenance_raw_id, meta)."""
    cands = [m for m in members if _rep_amount(m) is not None]
    if not cands:
        # capacity-only / no-amount fact: keep the provenance anchor's nulls
        return _amount_dict(canon), canon["raw_id"], \
            {"docs": 0, "groups": 0, "contested": False}

    groups: dict = {}
    for m in cands:
        key = (str(m.get("amount_currency") or "unknown"),
               repr(_round2sig(_rep_amount(m))))
        groups.setdefault(key, []).append(m)

    def prec(m):
        return _BOUND_PRECISION.get(m.get("amount_bound_type") or "", 0)

    def group_key(key):
        ms = groups[key]
        # Clean evidence first: a value group carried by at least one clean
        # member beats any group carried only by review_required / fallback /
        # audit-bad members, regardless of source rank.  Within that, the
        # documented CAN-1 order: source rank > bound precision > latest
        # doc_date > document support.
        return (max(int(_amount_clean(m)) for m in ms),
                max(_src(m) for m in ms),
                max(prec(m) for m in ms),
                max(_date(m) for m in ms),
                len({_doc_key(m) for m in ms}),
                str(key))

    win_key = max(groups, key=group_key)
    winner = max(groups[win_key],
                 key=lambda m: (int(_amount_clean(m)), prec(m), _src(m),
                                _date(m), m["raw_id"]))
    amount = _amount_dict(winner)
    # USD evidence is elected INSIDE the winning native-value group, but
    # independently of the native representation. A more precise/later member
    # may win the native amount while another member prints the source's USD
    # conversion; discarding the latter made valid conversions disappear at
    # fact canonicalization.
    usd_members = [m for m in groups[win_key]
                   if m.get("amount_usd_reported") is not None]
    usd_winner = max(usd_members,
                     key=lambda m: (_src(m), _date(m), m["raw_id"]),
                     default=None)
    if usd_winner is not None:
        amount["usd_reported"] = usd_winner.get("amount_usd_reported")
        amount["usd_reported_source"] = usd_winner.get("usd_reported_source")
        amount["usd_observation_id"] = usd_winner["raw_id"]
    meta = {"docs": len({_doc_key(m) for m in groups[win_key]}),
            "groups": len(groups),
            "contested": len(groups) > 1,
            "clean_winner": _amount_clean(winner),
            "usd_candidates_in_winning_group": len(usd_members),
            "usd_observation_id": (usd_winner or {}).get("raw_id")}
    return amount, winner["raw_id"], meta


def _elect_quantity(members: list[dict], canon: dict):
    """Elect the Fact's capacity instead of copying the provenance anchor's.

    The anchor-copy rule silently reported null capacity when the anchor
    lacked a quantity another member carried, and never flagged contested
    capacities.  Values are grouped on their MW-normalized magnitude
    (2 significant digits) so '2 GW' and '2,000 MW' are one value; ranking is
    document support > source rank > latest doc_date.  Falls back to the
    anchor's fields when no member carries a quantity."""
    cands = [m for m in members if m.get("quantity_value") is not None]
    if not cands:
        return (canon.get("quantity_value"), canon.get("quantity_unit"),
                {"docs": 0, "values": 0, "contested": False})
    groups: dict = {}
    for m in cands:
        mw = mw_value(m)
        key = (repr(_round2sig(mw)) if mw is not None
               else f"{m.get('quantity_value')}|{m.get('quantity_unit')}")
        groups.setdefault(key, []).append(m)

    def gkey(key):
        ms = groups[key]
        return (len({_doc_key(m) for m in ms}),
                max(_src(m) for m in ms),
                max(_date(m) for m in ms), str(key))

    win = max(groups, key=gkey)
    winner = max(groups[win], key=lambda m: (_src(m), _date(m), m["raw_id"]))
    meta = {"docs": len({_doc_key(m) for m in groups[win]}),
            "values": len(groups), "contested": len(groups) > 1}
    return winner.get("quantity_value"), winner.get("quantity_unit"), meta


def _series(members: list[dict], field: str) -> list:
    """Dated telling sequence [(doc_date, value), ...], informative values
    only, consecutive duplicates collapsed."""
    rows = sorted((m for m in members if m.get(field) not in _UNINFORMATIVE),
                  key=lambda m: (_date(m), m["raw_id"]))
    out = []
    for m in rows:
        v = m[field]
        if not out or out[-1][1] != v:
            out.append([_date(m), v])
    return out


def run(outdir: str) -> dict:
    man = Manifest(outdir, "build_fact_state")
    obs = read_jsonl(os.path.join(outdir, "observations_final.jsonl"))
    facts = read_jsonl(os.path.join(outdir, "facts.jsonl"))
    for name in ("observations_final.jsonl", "facts.jsonl"):
        man.add_input(os.path.join(outdir, name))
    obs_by_id = {r["raw_id"]: r for r in obs}

    states, participants = [], []
    n_contested = 0
    for f in facts:
        rids = f["raw_ids"]
        members = [obs_by_id[r] for r in rids]
        canon = obs_by_id[f["canonical_observation_id"]]
        amounts = []
        for m in members:
            if any(m.get(k) is not None for k in
                   ("amount_native_point", "amount_native_low", "amount_native_high")):
                amounts.append({"raw_id": m["raw_id"],
                                "point": m.get("amount_native_point"),
                                "low": m.get("amount_native_low"),
                                "high": m.get("amount_native_high"),
                                "currency": m.get("amount_currency"),
                                "bound_type": m.get("amount_bound_type"),
                                # AMT-2: the source's own USD conversion,
                                # with provenance -- never an FX invention.
                                "usd_reported": m.get("amount_usd_reported"),
                                "usd_reported_source": m.get("usd_reported_source"),
                                "amount_method": m.get("amount_method"),
                                "amount_flags": m.get("amount_flags") or [],
                                "observation_status": m.get("observation_status"),
                                "blocking_reasons": m.get("blocking_reasons") or [],
                                "basis": m.get("amount_basis"),
                                "aggregation_level": m.get("amount_aggregation_level"),
                                "doc_date": m.get("doc_date")})
        tset = _uniq(i for m in members for i in (m.get("target_entity_ids") or []))

        # ---- CAN-1 elections
        election = {}
        contested = []
        amount, amount_rid, amt_meta = _elect_amount(members, canon)
        election["amount"] = dict(amt_meta)
        if amt_meta["contested"]:
            contested.append("amount")
        elected = {}
        for col, field in _ELECTED_ENUMS:
            value, meta = _elect_enum(members, field)
            elected[col] = value
            election[field] = dict(meta)
            if meta["contested"]:
                contested.append(field)
        latest_status = max(
            (m for m in members
             if m.get("commitment_status") not in _UNINFORMATIVE),
            key=lambda m: (_date(m), _src(m), m["raw_id"]), default=None)
        # The name carrier applies the SAME audit disqualification as the
        # claim_source election itself: a member the election barred from
        # voting must not supply the canonical name either.
        claim_name_member = max(
            (m for m in members
             if m.get("claim_source_name")
             and m.get("claim_source") == elected["canonical_claim_source"]
             and not _audit_bad(m, "claim_source")),
            key=lambda m: (_src(m), _date(m), m["raw_id"]), default=None)
        quantity_value, quantity_unit, quantity_meta = \
            _elect_quantity(members, canon)
        election["quantity"] = dict(quantity_meta)
        if quantity_meta["contested"]:
            contested.append("quantity")
        if contested:
            n_contested += 1

        state = {
            "fact_id": f["fact_id"],
            "canonical_observation_id": f["canonical_observation_id"],
            "is_collective_target": f["is_collective_target"],
            "n_observations": len(rids),
            "target_entity_ids": tset,
            "target_canonicals": _uniq(c for m in members
                                       for c in (m.get("target_canonicals") or [])),
            "counterparty_entity_ids": _uniq(i for m in members
                                             for i in (m.get("counterparty_entity_ids") or [])),
            "parent_entity_ids": _uniq(i for m in members
                                       for i in (m.get("parent_entity_ids") or [])),
            "other_party_entity_ids": _uniq(i for m in members
                                            for i in (m.get("other_party_entity_ids") or [])),
            "object_ids": _uniq(m.get("object_id") for m in members),
            "object_canonicals": _uniq(m.get("object_canonical") for m in members),
            "parent_object_ids": _uniq(m.get("parent_object_id") for m in members),
            "parent_object_canonicals": _uniq(
                m.get("parent_object_canonical") for m in members),
            "object_types": _uniq(m.get("object_type") for m in members),
            "bases": _uniq(m.get("amount_basis") for m in members),
            "aggregation_levels": _uniq(
                m.get("amount_aggregation_level") for m in members),
            "currencies": _uniq(m.get("amount_currency") for m in members),
            "instrument_classes": _uniq(m.get("instrument_class") for m in members),
            "debt_instrument_subtypes": _uniq(
                m.get("debt_instrument_subtype") for m in members),
            "claim_sources": _uniq(m.get("claim_source") for m in members),
            "claim_source_names": _uniq(
                m.get("claim_source_name") for m in members),
            # ---- elected canonicals (CAN-1)
            "canonical_basis": elected["canonical_basis"],
            "canonical_aggregation_level": elected["canonical_aggregation_level"],
            "canonical_instrument": elected["canonical_instrument"],
            "canonical_debt_subtype": elected["canonical_debt_subtype"],
            "canonical_action": elected["canonical_action"],
            "canonical_claim_source": elected["canonical_claim_source"],
            "canonical_claim_source_name": (
                (claim_name_member or {}).get("claim_source_name")),
            "canonical_object_type": elected["canonical_object_type"],
            "canonical_period_scope": elected["canonical_period_scope"],
            "canonical_ai_scope": elected["canonical_ai_scope"],
            "canonical_geo_scope": elected["canonical_geo_scope"],
            "canonical_conditionality": elected["canonical_conditionality"],
            "canonical_amount": amount,
            "canonical_amount_observation_id": amount_rid,
            "canonical_usd_observation_id": amount.get("usd_observation_id"),
            # ---- serial status axes (never canonical)
            "commitment_status_latest": (latest_status or {}).get("commitment_status"),
            "commitment_status_series": _series(members, "commitment_status"),
            "deal_status_series": _series(members, "deal_status"),
            # ---- election transparency
            "election_meta": election,
            "contested_fields": contested,
            "quantity_value": quantity_value,
            "quantity_unit": quantity_unit,
            "amounts": amounts,
            "statuses": _uniq(m.get("observation_status") for m in members),
        }
        states.append(state)
        for m in members:
            for slot, ids in (("target", m.get("target_entity_ids")),
                              ("counterparty", m.get("counterparty_entity_ids")),
                              ("parent", m.get("parent_entity_ids")),
                              ("other_party", m.get("other_party_entity_ids"))):
                for eid in ids or []:
                    if eid:
                        participants.append({"fact_id": f["fact_id"],
                                             "raw_id": m["raw_id"],
                                             "entity_id": eid, "source_slot": slot})

    p1 = os.path.join(outdir, "fact_state.jsonl")
    p2 = os.path.join(outdir, "fact_participants.jsonl")
    write_jsonl(states, p1)
    write_jsonl(participants, p2)
    for p in (p1, p2):
        man.add_output(p)
    stats = {"facts": len(states), "participant_rows": len(participants),
             "facts_with_contested_field": n_contested}
    man.done(stats)
    return stats
