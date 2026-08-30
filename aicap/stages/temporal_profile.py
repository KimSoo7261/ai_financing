#!/usr/bin/env python3
"""Stage `build_temporal_profile` -- the authoritative per-Fact timeline.

Assembles each Fact's member observations into the shape the shared
projection engine (aicap.temporal_profile.build_profile) consumes, runs it,
and writes one profile row per Fact plus the consistency flags.

Active rules (all inside build_profile):
  * dropped-claim exclusion: a claim the time layer's resolver dropped
    (not_this_event) never enters the projection;
  * demotion: a member with a Stage-3/extraction time flag, or a claim with
    an uncertain escalation verdict, loses role-best ties to clean claims;
  * dated-before-dateless inside a role (a dateless winner would void it);
  * later-disclosure preference: inside a role, later doc_dates supersede;
  * duration-derived valid_to (a stated lease/PPA tenor closes a dated
    onset), with provenance recording whose observation supplied the tenor;
  * disclosure_closed eviction: an undated termination/cancellation closes
    the interval at that disclosure's quarter end;
  * fact-level firmness fallback from commitment_status when no dated onset
    exists;
  * quarter/year-anchored temporal_quality plus realized-overlap consistency
    flags.

Runs AFTER build_fact_state (which owns fact_state and participants).  This
stage is the sole producer of fact_temporal_profile.jsonl.

`members_of(outdir)` + `asof_profile(...)` give the view builder the
point-in-time variant (build_profile(asof=q_end)) over the same member shape.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

from ..temporal_profile import build_profile
from .common import Manifest, read_jsonl, write_jsonl


def members_of(outdir: str) -> tuple[dict, dict]:
    """Facts meta + {fact_id: member dicts} in the build_profile member shape."""
    obs = {r["raw_id"]: r for r in
           read_jsonl(os.path.join(outdir, "observations_final.jsonl"))}
    flags = {r["raw_id"]: r for r in
             read_jsonl(os.path.join(outdir, "obs_time_flags.jsonl"))}
    events_by_raw = defaultdict(list)
    for c in read_jsonl(os.path.join(outdir, "obs_time_claims.jsonl")):
        e = dict(c)
        # The claim table stores the id as claim_id; the profile layer reads
        # it back as temporal_event_id.
        e["temporal_event_id"] = e.pop("claim_id", None)
        events_by_raw[str(e.get("raw_id"))].append(e)
    facts = {}
    members = {}
    for f in read_jsonl(os.path.join(outdir, "facts.jsonl")):
        facts[f["fact_id"]] = {
            "canonical_observation_id": f["canonical_observation_id"],
            "is_collective_target": f.get("is_collective_target")}
        rows = []
        for rid in f["raw_ids"]:
            o = obs.get(rid, {})
            fl = flags.get(rid, {})
            rows.append({
                "raw_id": rid,
                "doc_date": o.get("doc_date"),
                "commitment_status": o.get("commitment_status"),
                "events": events_by_raw.get(rid, []),
                "time_flagged": bool(o.get("audit_time_bad")
                                     or fl.get("time_needs_review")),
            })
        members[f["fact_id"]] = rows
    return facts, members


def project(prof: dict) -> dict:
    """Add the onset/offset convenience fields the views read."""
    out = dict(prof)
    if prof.get("valid_from_best"):
        out["onset"] = {"role": prof.get("valid_from_basis"),
                        "date_start": prof.get("valid_from_best"),
                        "date_end": None,
                        "granularity": prof.get("valid_from_granularity"),
                        # The economic gate reads firmness_best, which includes
                        # the commitment_status fallback.
                        "firmness": prof.get("firmness_best")}
    else:
        out["onset"] = None
    if prof.get("valid_to_best"):
        out["offset"] = {"date_end": prof.get("valid_to_best"),
                         "basis": prof.get("valid_to_basis"),
                         "firmness": prof.get("firmness_best"),
                         "granularity": None}
    else:
        out["offset"] = None
    out["last_disclosed_at"] = prof.get("last_updated_at")
    return out


def asof_profile(fact_id: str, meta: dict, members: list, q_end: str) -> dict:
    prof, _flags = build_profile(fact_id, meta["canonical_observation_id"],
                                 members, asof=q_end,
                                 collective=meta.get("is_collective_target"))
    return project(prof)


def run(outdir: str) -> dict:
    man = Manifest(outdir, "build_temporal_profile")
    for name in ("facts.jsonl", "observations_final.jsonl",
                 "obs_time_claims.jsonl", "obs_time_flags.jsonl"):
        man.add_input(os.path.join(outdir, name))

    facts, members = members_of(outdir)
    profiles, all_flags = [], []
    tally = Counter()
    for fid in sorted(facts):
        prof, flags = build_profile(fid, facts[fid]["canonical_observation_id"],
                                    members[fid],
                                    collective=facts[fid]["is_collective_target"])
        profiles.append(project(prof))
        all_flags.extend(flags)
        tally[f"quality_{prof['temporal_quality']}"] += 1
        if prof.get("valid_from_best"):
            tally["with_onset"] += 1
        if prof.get("valid_to_basis") == "duration_derived":
            tally["valid_to_duration_derived"] += 1
        if prof.get("valid_to_basis") == "disclosure_closed":
            tally["valid_to_disclosure_closed"] += 1
        if prof.get("onset_from_time_flagged"):
            tally["onset_from_time_flagged"] += 1
        if prof.get("n_time_flagged_members"):
            tally["facts_with_flagged_members"] += 1
        if prof.get("firmness_basis") == "commitment_status":
            tally["firmness_from_status_fallback"] += 1

    p1 = os.path.join(outdir, "fact_temporal_profile.jsonl")
    p2 = os.path.join(outdir, "fact_profile_flags.jsonl")
    write_jsonl(profiles, p1)
    write_jsonl(all_flags, p2)
    man.add_output(p1, len(profiles))
    man.add_output(p2, len(all_flags))
    stats = {"facts": len(profiles), "consistency_flags": len(all_flags),
             **dict(tally)}
    man.done(stats)
    return stats
