#!/usr/bin/env python3
"""The authoritative per-Fact temporal projection: build_profile.

One profile row per Fact: the per-role best time claim, valid_from/valid_to
with their basis, first/last disclosure dates, a fact-level firmness, and a
temporal_quality grade.  Pure projection -- rebuildable from observations at
any time; Facts themselves stay identity-only.

Claims the time layer DROPPED (escalation verdict not_this_event) never
enter the projection at all -- kept in obs_time_claims as evidence only.
The ranking that picks each role's best claim (see _claim_sort_key):

  1. demotion -- a claim from a time-flagged observation, or one whose
     escalation verdict was uncertain/quarantined, loses to ANY clean claim
     and wins a role only when no clean claim exists for it;
  2. dated before dateless -- role_best exists to supply dates to the
     onset/offset picks; a dateless winner would silently void its role;
  3. firmness (realized > committed > prospective) -- firmness outranks
     precision: a realized month beats a prospective day;
  4. granularity (finer wins within one firmness);
  5. LATER doc_date wins ties (later disclosures supersede);
  6. raw_id as the deterministic tiebreak.

Onset (valid_from) walks roles in priority order bind > build > operate >
commit and takes the first role whose best claim has a date_start.  Offset
(valid_to) is the end-role date when one exists; else a stated tenor closes a
dated onset (duration_derived, with a provenance link recording whether the
tenor came from the onset observation or another one); else an undated
termination/cancellation closes the interval at that disclosure's quarter end
(disclosure_closed).

profile_asof semantics: build_profile(asof=q_end) recomputes the WHOLE
projection using only observations disclosed by q_end -- the point-in-time
variant the known views must use (no look-ahead).
"""
from __future__ import annotations

from . import time_rules as TR

FIRMNESS_ORDER = ["realized", "committed", "prospective"]
GRANULARITY_ORDER = ["day", "month", "quarter", "fiscal_quarter", "half_year",
                     "year", "fiscal_year", "multi_year", "unknown"]
# onset priority: legally bound > build begins > operations begin > announced
ONSET_ROLES = ["bind", "build", "operate", "commit"]
OFFSET_ROLES = ["end"]

_ROLES = ("commit", "bind", "build", "operate", "end", "period")
_FIRM_RANK = {m: i for i, m in enumerate(FIRMNESS_ORDER)}
_GRAN_RANK = {g: i for i, g in enumerate(GRANULARITY_ORDER)}

# Fact-level firmness fallback for Facts with NO dated onset:
# commitment_status is a Stage-2 claim, not a time claim, so using it here is
# clearly labelled (firmness_basis="commitment_status").  Strongest across
# members wins: realized > committed > prospective.
_STATUS_FIRMNESS = {"operational": "realized", "under_construction": "realized",
                    "contracted": "committed", "approved": "committed",
                    "announced": "prospective", "planned": "prospective",
                    "proposed": "prospective"}
_FIRMNESS_STRENGTH = {"realized": 0, "committed": 1, "prospective": 2}


def _norm_event(e):
    """One claim dict -> the normalized shape build_profile ranks.  Off-
    vocabulary role/firmness values degrade safely (role None never wins a
    role slot; firmness falls back to prospective)."""
    ds, de = e.get("date_start"), e.get("date_end")
    raw_role = e.get("role")
    raw_firm = e.get("firmness")
    return {
        "role": raw_role if raw_role in _ROLES else None,
        "firmness": raw_firm if raw_firm in _FIRM_RANK else "prospective",
        "granularity": e.get("granularity") or "unknown",
        "interval": e.get("interval"),
        "date_start": ds, "date_end": de,
        "calendar_quarter": e.get("calendar_quarter") or TR.calendar_quarter_of(ds, de),
        "claim_status": e.get("claim_status") or "active",
        "screen_flags": e.get("screen_flags") or [],
        "escalation_verdict": e.get("escalation_verdict"),
        "temporal_event_id": e.get("temporal_event_id"),
        "raw_id": e.get("raw_id"),
    }


def _claim_sort_key(claim, doc_date):
    """Smaller is better: demoted claims last (a member time flag or an
    `uncertain`/quarantined escalation verdict -- such a claim wins a role
    only when no clean claim exists for it), then DATED before dateless
    (role_best exists to supply dates to the onset/offset picks; a dateless
    winner used to silently void its whole role -- 2026-08-19 revision), then
    firmness rank, granularity rank, LATER doc_date (later disclosures
    supersede), then raw_id for determinism."""
    return (int(bool(claim.get("time_flagged"))
                or claim.get("claim_status") == "demoted"),
            int(not (claim.get("date_start") or claim.get("date_end"))),
            _FIRM_RANK.get(claim["firmness"], len(FIRMNESS_ORDER)),
            _GRAN_RANK.get(claim["granularity"], len(GRANULARITY_ORDER)),
            "￿" if doc_date is None else _invert_date(doc_date),
            claim["raw_id"] or "")


def _invert_date(iso):
    """Lexicographic inversion so LATER dates sort FIRST inside min()."""
    return "".join(chr(255 - ord(c)) for c in str(iso))


def _quarter_end_of(doc_date):
    try:
        y, q = TR.quarter_of_date(str(doc_date)[:10])
    except (ValueError, TypeError):
        return None
    return TR.quarter_bounds(y, q)[1]


def _envelope_days(claim):
    import datetime as dt
    try:
        a = dt.date.fromisoformat(claim["date_start"])
        b = dt.date.fromisoformat(claim["date_end"])
    except (TypeError, ValueError):
        return None
    return (b - a).days


_DUR_MONTHS = {"year": 12, "quarter": 3, "month": 1}
_DUR_DAYS = {"week": 7, "day": 1}


def _best_duration(members):
    """Longest stated tenor across the Fact's claims (a commitment's full
    coverage span, e.g. a 20-year PPA), as (value, unit, source_raw_id) or
    None.  The source raw_id is kept so the derived valid_to can carry a
    duration LINK provenance -- a tenor read from another observation than
    the onset claim's may belong to a different tranche/lease."""
    best, best_months = None, -1.0
    for m in members:
        for e in m.get("events") or []:
            v, u = e.get("duration_value"), e.get("duration_unit")
            if not v or not u:
                continue
            months = v * _DUR_MONTHS.get(u, 0) + v * _DUR_DAYS.get(u, 0) / 30.0
            if months > best_months:
                best, best_months = (int(v), u, m.get("raw_id")), months
    return best


def _add_duration(iso, value, unit):
    """iso + duration -> inclusive end date (month-clamped), or None."""
    import calendar as _cal
    import datetime as dt
    try:
        d = dt.date.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return None
    if unit in _DUR_DAYS:
        return (d + dt.timedelta(days=value * _DUR_DAYS[unit])).isoformat()
    months = value * _DUR_MONTHS.get(unit, 0)
    if not months:
        return None
    m0 = d.month - 1 + months
    y, m = d.year + m0 // 12, m0 % 12 + 1
    day = min(d.day, _cal.monthrange(y, m)[1])
    return (dt.date(y, m, day) - dt.timedelta(days=1)).isoformat()


def build_profile(fact_id, canonical_id, members, asof=None, collective=None):
    """members: list of {raw_id, doc_date, commitment_status, events:[claim
    dicts], time_flagged?}.  asof: ISO date -- use only members with
    doc_date <= asof (point-in-time mode).  collective: the Fact's
    is_collective_target marker, passed through so consumers never attribute
    a collective Fact's timeline to a single member firm.
    Returns (profile_row, consistency_flags)."""
    if asof is not None:
        members = [m for m in members if m.get("doc_date") and str(m["doc_date"]) <= str(asof)]
    doc_dates = sorted(str(m["doc_date"]) for m in members if m.get("doc_date"))
    claims = []
    for m in members:
        for e in m.get("events") or []:
            # A claim the resolver DROPPED (not_this_event -- e.g. a
            # participant's founding date) stays in obs_time_claims as
            # evidence but never enters the temporal projection: the fact
            # falls back to disclosure_only rather than carrying a date that
            # belongs to something else (2026-08-19 downgrade-never-block).
            if (e.get("claim_status") or "active") == "dropped":
                continue
            c = _norm_event(e)
            c["doc_date"] = m.get("doc_date")
            c["time_flagged"] = bool(m.get("time_flagged"))
            claims.append(c)

    # per-role best claim
    role_best = {}
    for c in claims:
        role = c["role"]
        if not role:
            continue
        cur = role_best.get(role)
        if cur is None or _claim_sort_key(c, c["doc_date"]) < _claim_sort_key(cur, cur["doc_date"]):
            role_best[role] = c

    def _pick(roles, date_field):
        for role in roles:
            c = role_best.get(role)
            if c and c.get(date_field):
                return role, c
        return None, None

    vf_basis, vf_claim = _pick(ONSET_ROLES, "date_start")
    vt_basis, vt_claim = _pick(OFFSET_ROLES, "date_end")
    valid_from = vf_claim["date_start"] if vf_claim else None
    valid_to = vt_claim["date_end"] if vt_claim else None

    # A dated onset + a stated tenor (lease/PPA years) closes the interval
    # deterministically; a real end-role date still wins above.
    duration_link = None
    if valid_to is None and valid_from:
        dur = _best_duration(members)
        if dur:
            value, unit, dur_raw = dur
            derived = _add_duration(valid_from, value, unit)
            if derived and derived >= valid_from:
                valid_to = derived
                vt_basis = "duration_derived"
                duration_link = ("onset_observation"
                                 if vf_claim and str(dur_raw) == str(vf_claim.get("raw_id"))
                                 else "other_observation")

    # Eviction: an undated termination/cancellation closes the interval at
    # that observation's disclosure quarter-end ("we only know it died
    # sometime by then").
    if valid_to is None:
        closed = sorted(str(m["doc_date"]) for m in members
                        if (m.get("commitment_status") in ("terminated", "cancelled"))
                        and m.get("doc_date"))
        if closed:
            valid_to = _quarter_end_of(closed[0])
            vt_basis = "disclosure_closed"

    # Fact-level firmness -- the dated onset's firmness when one exists, else
    # the strongest commitment_status stage across members.
    onset_firm = vf_claim["firmness"] if vf_claim else None
    status_firm = None
    for m in members:
        f = _STATUS_FIRMNESS.get(str(m.get("commitment_status") or "").lower())
        if f and (status_firm is None
                  or _FIRMNESS_STRENGTH[f] < _FIRMNESS_STRENGTH[status_firm]):
            status_firm = f
    firmness_best = onset_firm or status_firm
    firmness_basis = ("onset_claim" if onset_firm
                      else ("commitment_status" if status_firm else None))

    # temporal_quality from the ONSET claim's envelope
    if vf_claim and vf_claim["calendar_quarter"]:
        quality = "quarter_anchored"
    elif vf_claim and (d := _envelope_days(vf_claim)) is not None and d <= 366:
        quality = "year_anchored"
    elif doc_dates:
        quality = "disclosure_only"
    else:
        quality = "unresolved"

    # consistency: same role, both realized, both dated, disjoint envelopes
    flags = []
    by_role = {}
    for c in claims:
        if (c["role"] and c["firmness"] == "realized"
                and c["date_start"] and c["date_end"]):
            by_role.setdefault(c["role"], []).append(c)
    for role, group in by_role.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["date_end"] < b["date_start"] or b["date_end"] < a["date_start"]:
                    flags.append({"fact_id": fact_id, "role": role,
                                  "raw_ids": sorted({a["raw_id"], b["raw_id"]}),
                                  "detail": f'{role}: {a["date_start"]}..{a["date_end"]} '
                                            f'vs {b["date_start"]}..{b["date_end"]}'})

    profile = {
        "fact_id": fact_id,
        "canonical_observation_id": canonical_id,
        "n_members": len(members),
        "n_temporal_events": len(claims),
        "first_disclosed_at": doc_dates[0] if doc_dates else None,
        "last_updated_at": doc_dates[-1] if doc_dates else None,
        "valid_from_best": valid_from,
        "valid_from_basis": vf_basis,
        "valid_from_quarter": vf_claim["calendar_quarter"] if vf_claim else None,
        "valid_from_modality": vf_claim["firmness"] if vf_claim else None,
        "valid_from_granularity": vf_claim["granularity"] if vf_claim else None,
        "valid_to_best": valid_to,
        "valid_to_basis": vt_basis,
        "valid_to_duration_link": duration_link,
        "firmness_best": firmness_best,
        "firmness_basis": firmness_basis,
        "temporal_quality": quality,
        "roles_present": ";".join(sorted(role_best)),
        "n_consistency_flags": len(flags),
        "is_collective_target": collective,
        "n_time_flagged_members": sum(1 for m in members if m.get("time_flagged")),
        "onset_from_time_flagged": bool(vf_claim and vf_claim.get("time_flagged")),
        "role_best": {r: {k: v for k, v in c.items() if k != "doc_date"}
                      for r, c in sorted(role_best.items())},
    }
    return profile, flags
