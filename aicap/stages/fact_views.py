#!/usr/bin/env python3
"""Stage `build_fact_views` -- the quarterly Fact views.

The pipeline's data model stops at the Fact layer; these views are its final
products, built directly from Facts and their evidence.  Two notions of time
are kept strictly apart: the KNOWN views see only what was disclosed by each
quarter's end (no look-ahead, enforced by invariants), while the ECONOMIC
view uses the ex-post best knowledge of when commitments actually happened.

Per quarter, into <outdir>/fact_views/:
  known_facts_<q>              as-of information set, one row per known Fact
  fact_disclosure_events_<q>   fact_first_disclosed events (+ per-family,
                               per-currency masses in the report)
  economic_facts_<q>           ex-post fine-granularity + realized onset gate
  target_quarter_measures_<q>  target-set x family x basis x aggregation-level
                               x currency x scope
                               cells; a single Fact yields a value, several
                               competing Facts FAIL CLOSED (value null +
                               candidates listed); collective targets stay
                               one hyperedge key, never fanned out
  fact_network_<q>             Fact x participant evidence rows
"""
from __future__ import annotations

import os
from collections import defaultdict

from .common import (Manifest, SOURCE_RANK, family_of, mw_value,
                     q_bounds, q_of, q_range, read_jsonl,
                     representative_amount, write_jsonl, write_json)
from .temporal_profile import asof_profile, members_of

from ..config import flow_direction_of_instrument

FINE_ONSET = {"day", "month", "quarter"}


def _asof_source(members: list[dict], canonical_id: str, qe: str):
    """Which observation supplies a Fact's amount fields in quarter qe: the
    canonical observation if already disclosed by qe, else the best dated
    member (source rank, then latest doc_date, then raw_id).  A Fact's
    canonical member may post-date early quarters -- this substitution keeps
    the known view honest."""
    by_id = {m["raw_id"]: m for m in members}
    canon = by_id.get(canonical_id)
    # if canonical is before qe
    if canon and str(canon.get("doc_date") or "9999") <= qe:
        return canonical_id, False
    # if canonical is after qe then filter all the members before qe
    dated = [m for m in members if str(m.get("doc_date") or "9999") <= qe]
        # if none, then this fact shouldn't be in the known set for this quarter
    if not dated:
        return None, False
        # else: select the best, source rank > newest doc date > raw id
    best = max(dated, key=lambda m: (SOURCE_RANK.get(m.get("source_type"), 1),
                                     str(m.get("doc_date") or ""), m["raw_id"]))
    return best["raw_id"], best["raw_id"] != canonical_id


def run(outdir: str, include_projected: bool = False) -> dict:
    man = Manifest(outdir, "build_fact_views")
    obs = read_jsonl(os.path.join(outdir, "observations_final.jsonl"))
    facts = read_jsonl(os.path.join(outdir, "facts.jsonl"))
    profs = {
        p["fact_id"]: p
        for p in read_jsonl(
            os.path.join(outdir, "fact_temporal_profile.jsonl")
        )
    }
    parts = read_jsonl(os.path.join(outdir, "fact_participants.jsonl"))
    # obs_time_flags/obs_time_claims are read inside members_of below; they
    # are declared as inputs here so the manifest lists every file this stage
    # actually opens.
    for name in ("observations_final.jsonl", "facts.jsonl",
                 "fact_temporal_profile.jsonl", "fact_participants.jsonl",
                 "obs_time_claims.jsonl", "obs_time_flags.jsonl"):
        man.add_input(os.path.join(outdir, name))

    obs_by_id = {r["raw_id"]: r for r in obs}
    parts_by_fact = defaultdict(list)
    for p in parts:
        parts_by_fact[p["fact_id"]].append(p)
    facts_by = {f["fact_id"]: f for f in facts}

    # As-of profiles are full recomputations of build_profile over the
    # members disclosed by each quarter end -- not a filter on the final
    # profile.
    pmeta, pmembers = members_of(outdir)

    disc_q = {f: q_of(profs[f].get("first_disclosed_at")) for f in facts_by}
    quarters = q_range([q for q in disc_q.values() if q] +
                       [q_of(r.get("doc_date")) for r in obs if q_of(r.get("doc_date"))])
    outroot = os.path.join(outdir, "fact_views")
    report = {"quarters": {}, "include_projected": include_projected}
    violations = defaultdict(list)
    prev_known: set = set()

    for q in quarters:
        qs, qe = q_bounds(q)
        known_rows, event_rows, econ_rows, cell_rows, net_rows = [], [], [], [], []
        mass: dict = {}
        # Facts first disclosed by this quarter's end; a Fact never leaves
        # the known set once it enters (monotonicity is asserted below).
        known_fids = sorted(f for f in facts_by if (disc_q.get(f) or "9999Q9") <= q)

        # ---- known view (as-of / ex-ante: only evidence disclosed by qe)
        for fid in known_fids:
            f = facts_by[fid]
            members = [obs_by_id[r] for r in f["raw_ids"]]
            dated_ids = sorted(r["raw_id"] for r in members
                               if str(r.get("doc_date") or "9999") <= qe)
            src_id, substituted = _asof_source(members, f["canonical_observation_id"], qe)
            if src_id is None:
                violations["coverage"].append((q, fid, "no_dated_member"))
                continue
            src = obs_by_id[src_id]
            if str(src.get("doc_date") or "9999") > qe:
                violations["asof_source_dates"].append((q, fid))
            # the full temporal projection, recomputed as of quarter end
            ap = asof_profile(fid, pmeta[fid], pmembers[fid], qe)
            tgt = sorted({t for r in dated_ids
                          for t in (obs_by_id[r].get("target_entity_ids") or []) if t})
            cp = sorted({t for r in dated_ids
                         for t in (obs_by_id[r].get("counterparty_entity_ids") or []) if t})
            par = sorted({t for r in dated_ids
                          for t in (obs_by_id[r].get("parent_entity_ids") or []) if t})
            obj = sorted({obs_by_id[r].get("object_id") for r in dated_ids
                          if obs_by_id[r].get("object_id")})
            pobj = sorted({obs_by_id[r].get("parent_object_id") for r in dated_ids
                           if obs_by_id[r].get("parent_object_id")})
            obj_types = sorted({obs_by_id[r].get("object_type") for r in dated_ids
                                if obs_by_id[r].get("object_type")})
            amount = representative_amount(src)
            family = family_of(src.get("amount_basis"))
            newly = disc_q.get(fid) == q
            onset = (ap.get("onset") or {})
            row = {
                "quarter": q, "view": "known_fact", "fact_id": fid,
                "known_raw_ids": dated_ids,
                "n_known_observations": len(dated_ids),
                "newly_disclosed": newly,
                "asof_source_raw_id": src_id,
                "asof_member_substituted": substituted,
                "amount_native": amount,
                "amount_currency": src.get("amount_currency"),
                "amount_basis": src.get("amount_basis"),
                "amount_bound_type": src.get("amount_bound_type"),
                "amount_aggregation_level": src.get("amount_aggregation_level"),
                "amount_period_scope": src.get("amount_period_scope"),
                "measure_family": family,
                "instrument_class": src.get("instrument_class"),
                "debt_instrument_subtype": src.get("debt_instrument_subtype"),
                "claim_source": src.get("claim_source"),
                "conditionality": src.get("conditionality"),
                "commitment_status": src.get("commitment_status"),
                "ai_infra_scope": src.get("ai_infra_scope"),
                "flow_direction": flow_direction_of_instrument(src.get("instrument_class")),
                "quantity_mw": mw_value(src),
                "target_entity_ids_asof": tgt,
                "counterparty_entity_ids_asof": cp,
                "parent_entity_ids_asof": par,
                "object_ids_asof": obj,
                "parent_object_ids_asof": pobj,
                "object_types_asof": obj_types,
                "is_collective_target": f.get("is_collective_target"),
                "asof_onset_role": onset.get("role"),
                "asof_onset_date": onset.get("date_start"),
                "asof_onset_firmness": onset.get("firmness"),
                "asof_temporal_quality": ap.get("temporal_quality"),
                "expost_n_observations": len(members),
                "expost_all_disclosed": len(dated_ids) == len(members),
            }
            known_rows.append(row)
            
            # fact_disclosure_events
            if newly:
                event_rows.append({
                    "quarter": q, "view": "fact_disclosure_event",
                    "event_type": "fact_first_disclosed", "fact_id": fid,
                    "first_disclosed_at": profs[fid].get("first_disclosed_at"),
                    "amount_native": amount,
                    "amount_currency": src.get("amount_currency"),
                    "amount_basis": src.get("amount_basis"),
                    "amount_bound_type": src.get("amount_bound_type"),
                    "amount_aggregation_level": src.get("amount_aggregation_level"),
                    "measure_family": family, "quantity_mw": mw_value(src),
                    "is_collective_target": f.get("is_collective_target"),
                    "target_entity_ids_asof": tgt,
                })
                # A fact whose first-disclosure quarter equals q emits a
                # fact_first_disclosed event, and its amount accumulates into
                # the quarter's GROSS disclosure mass, keyed by measure-family
                # x amount_basis x currency.  Basis in the key stops a deal's
                # financing_total and its debt/equity/grant components (four
                # bases, ONE family) from double counting in one family sum;
                # unit rates are excluded (a $/MW rate is never a total);
                # bounds enter at their stated value, so the mass stays a
                # labeled GROSS diagnostic, never a net aggregate.
                level = src.get("amount_aggregation_level") or "unknown"
                if (family != "capacity" and amount is not None
                        and level != "unit_rate"):
                    fam = mass.setdefault(family, {})
                    bas = fam.setdefault(src.get("amount_basis") or "unknown", {})
                    cur = src.get("amount_currency") or "unknown"
                    bas[cur] = bas.get(cur, 0.0) + float(amount)
            for p in sorted(parts_by_fact.get(fid, []),
                            key=lambda x: (x["raw_id"], x["source_slot"], x["entity_id"])):
                if str(obs_by_id[p["raw_id"]].get("doc_date") or "9999") > qe:
                    continue
                net_rows.append({
                    "quarter": q, "view": "fact_network", "fact_id": fid,
                    "entity_id": p["entity_id"], "source_slot": p["source_slot"],
                    "source_raw_id": p["raw_id"],
                    "flow_direction": row["flow_direction"],
                    "measure_family": family,
                    "is_collective_target": f.get("is_collective_target"),
                })

        # ---- economic view (ex-post): a Fact is economically active in q
        # iff its onset is dated, FINE-grained (day/month/quarter -- a
        # year-blurred onset is never pinned to a quarter), REALIZED, and the
        # validity interval overlaps the quarter.  include_projected=True
        # additionally admits fine-but-unrealized onsets, marked projected.
        for fid, p in sorted(profs.items()):
            onset, offset = p.get("onset") or {}, p.get("offset") or {}
            vf = onset.get("date_start")
            if not vf:
                continue
            fine = onset.get("granularity") in FINE_ONSET
            realized = onset.get("firmness") == "realized"
            admit = fine and realized
            projected = fine and not realized and include_projected
            vt = offset.get("date_end")
            # must have intersection(start, end) vs (q start, q end), 
            # [vf,vt] ∩ [qs,qe] ≠ ∅
            if (admit or projected) and vf <= qe and (not vt or vt >= qs):
                econ_rows.append({
                    "quarter": q, "view": "economic_fact", "fact_id": fid,
                    "economic_active": True, "valid_from": vf, "valid_to": vt,
                    "valid_from_granularity": onset.get("granularity"),
                    "firmness": onset.get("firmness"), "projected": projected,
                    "onset_in_quarter": q_of(vf) == q,
                    "is_collective_target": facts_by[fid].get("is_collective_target"),
                })

        # ---- fail-closed target cells
        cells = defaultdict(list)
        for row in known_rows:
            tkey = ";".join(row["target_entity_ids_asof"])
            if not tkey:
                continue
            if row["measure_family"] == "capacity":
                if row["quantity_mw"] is not None:
                    cells[(tkey, "capacity", "capacity_only",
                           row.get("amount_aggregation_level") or "unknown",
                           "MW", "na")].append(
                        (row["fact_id"], row["quantity_mw"], row))
                continue
            if row["amount_native"] is None:
                continue
            cells[(tkey, row["measure_family"], row["amount_basis"] or "unknown",
                   row.get("amount_aggregation_level") or "unknown",
                   row["amount_currency"] or "unknown",
                   row.get("amount_period_scope") or "unclear")].append(
                (row["fact_id"], row["amount_native"], row))
        for (tkey, fam, basis, level, cur, scope), mem in sorted(cells.items()):
            if fam == "unclassified":
                status, value = "unclassified_excluded", None
            elif level == "unknown":
                status, value = "aggregation_level_unknown", None
            elif level == "unit_rate":
                status, value = "unit_rate_excluded", None
            elif fam in {"valuation", "project_attribute", "program_attribute"}:
                status, value = "non_additive_basis", None
            elif len(mem) == 1:
                status, value = "single_fact", mem[0][1]
            else:
                status, value = "multi_fact_unresolved", None
            # Aggregation guards a reader must see BEFORE quoting the value:
            # a "single_fact" number can still be an upper bound, conditional,
            # merely proposed, or only partly AI-allocated.  For a single-fact
            # cell they describe that fact; for multi-fact cells the sets show
            # what the unresolved candidates carry.
            src_rows = [r for _f, _v, r in mem]
            bounds = sorted({r.get("amount_bound_type") or "unclear"
                             for r in src_rows})
            conds = sorted({r.get("conditionality") or "unclear"
                            for r in src_rows})
            commits = sorted({str(r.get("commitment_status") or "unclear")
                              for r in src_rows})
            scopes_ai = sorted({r.get("ai_infra_scope") or "unclear"
                                for r in src_rows})
            claims = sorted({r.get("claim_source") or "unknown"
                             for r in src_rows})
            cell_rows.append({
                "quarter": q, "view": "target_quarter_measure",
                "target_key": tkey, "n_target_entities": len(tkey.split(";")),
                "measure_family": fam, "amount_basis": basis, "currency": cur,
                "amount_aggregation_level": level,
                "period_scope": scope, "value": value, "status": status,
                "value_bound_type": (bounds[0] if status == "single_fact"
                                     and len(bounds) == 1 else None),
                "bound_types": bounds,
                "conditionality": conds,
                "commitment_status": commits,
                "ai_infra_scope": scopes_ai,
                "claim_sources": claims,
                "candidate_fact_ids": sorted(f for f, _v, _r in mem),
                "candidate_values": [v for _f, v, _r in
                                     sorted(mem, key=lambda t: (t[0], t[1]))],
            })

        now_known = {r["fact_id"] for r in known_rows}
        if not prev_known <= now_known:
            violations["monotone_known"].append((q,))
        prev_known = now_known
        for c in cell_rows:
            if len(c["candidate_fact_ids"]) > 1 and c["value"] is not None:
                violations["fail_closed"].append((q, c["target_key"]))

        for name, view_rows in (
                (f"known_facts_{q}.jsonl", known_rows),
                (f"fact_disclosure_events_{q}.jsonl", event_rows),
                (f"economic_facts_{q}.jsonl", econ_rows),
                (f"target_quarter_measures_{q}.jsonl", cell_rows),
                (f"fact_network_{q}.jsonl", net_rows)):
            path = os.path.join(outroot, name)
            write_jsonl(view_rows, path)
            man.add_output(path, len(view_rows))
        report["quarters"][q] = {
            "known_facts": len(known_rows),
            "newly_disclosed": sum(1 for r in known_rows if r["newly_disclosed"]),
            "economic_facts": len(econ_rows),
            "measure_cells": len(cell_rows),
            "measure_cells_unresolved": sum(1 for c in cell_rows
                                            if c["status"] == "multi_fact_unresolved"),
            "network_rows": len(net_rows),
            # GROSS by construction: newly disclosed facts' stated values per
            # family x basis x currency; bounds at stated value, no netting.
            "disclosure_new_fact_mass_gross": mass,
        }

    report["invariants"] = {k: {"ok": False, "violations": v[:20]}
                            for k, v in violations.items()}
    for name in ("asof_source_dates", "coverage", "monotone_known",
                 "fail_closed"):
        report["invariants"].setdefault(name, {"ok": True, "violations": []})
    report["success"] = all(c["ok"] for c in report["invariants"].values())
    report_path = os.path.join(outroot, "fact_views_report.json")
    write_json(report, report_path)
    man.add_output(report_path)

    stats = {"quarters": {q: {k: v for k, v in s.items()
                              if k != "disclosure_new_fact_mass_gross"}
                          for q, s in report["quarters"].items()},
             "success": report["success"]}
    man.done(stats)
    if not report["success"]:
        raise SystemExit("build_fact_views invariants failed "
                         f"({sorted(k for k, v in report['invariants'].items() if not v['ok'])})")
    return stats
