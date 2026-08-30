#!/usr/bin/env python3
"""Stage `intake` -- Stage-3 output -> immutable observation store.

Reads capital_instances.jsonl + audit.jsonl from the source run, verifies the
hard preconditions (unique raw_id, quote present, valid doc_date), attaches
audit verdicts as intake flags, and writes observations.jsonl verbatim-plus-
flags.  Nothing here interprets economics; failures quarantine, never mutate.
"""
from __future__ import annotations

import os

from .common import (Manifest, is_nullish, parse_date, read_jsonl,
                     write_jsonl)
from ..config import SCHEMA_VERSION

AUDIT_CHECKS = ["in_scope", "anchor_direction", "amount_ok", "amount_basis_ok",
                "instrument_ok", "object_ok", "ai_scope_ok", "time_ok"]


def run(indir: str, outdir: str) -> dict:
    man = Manifest(outdir, "intake")
    inst_path = os.path.join(indir, "capital_instances.jsonl")
    audit_path = os.path.join(indir, "audit.jsonl")
    man.add_input(inst_path)
    man.add_input(audit_path)

    instances = read_jsonl(inst_path)
    incompatible = [
        {"raw_id": row.get("raw_id"),
         "schema_version": row.get("schema_version") or "legacy_unversioned"}
        for row in instances
        if row.get("schema_version") != SCHEMA_VERSION
    ]
    if incompatible:
        versions = sorted({row["schema_version"] for row in incompatible})
        raise RuntimeError(
            "intake schema gate: expected " + SCHEMA_VERSION
            + f", found {versions} on {len(incompatible)}/{len(instances)} rows; "
            "rerun extract+audit under pipeline_new. Legacy amount-basis "
            "crosswalks are documentation aids, not permission to guess the "
            "new object/aggregation/source fields downstream."
        )
    audits: dict = {}
    n_dup_audit = 0
    for a in read_jsonl(audit_path):
        rid = str(a.get("raw_id") or "")
        if not rid:
            continue
        if rid in audits:
            n_dup_audit += 1        # last-win, but counted -- a duplicate
        audits[rid] = a             # audit raw_id signals an upstream bug
    if n_dup_audit:
        print(f"  intake WARNING: {n_dup_audit} duplicate audit raw_id(s) "
              f"(last row wins) -- inspect audit.jsonl")

    seen, rows, quarantine = set(), [], []
    for src in instances:
        row = dict(src)
        rid = str(row.get("raw_id") or "")
        bad = []
        if not rid:
            bad.append("missing_raw_id")
        elif rid in seen:
            bad.append("duplicate_raw_id")
        if is_nullish(row.get("verbatim_quote")):
            bad.append("missing_verbatim_quote")
        if not is_nullish(row.get("doc_date")) and parse_date(row.get("doc_date")) is None:
            bad.append("invalid_doc_date")
        if bad:
            quarantine.append({"raw_id": rid or None, "reasons": bad, "row": src})
            continue
        seen.add(rid)
        audit = audits.get(rid)
        flags = {"audit_present": audit is not None,
                 "audit_bad": [], "audit_unsure": [], "audit_caveat": [],
                 "quote_verbatim_pass": None,
                 # GATE-1': graded verbatim (pass|drift|reorder|fatal) + the
                 # missing-token evidence; None on audits written before the
                 # grade existed (the gate then falls back to the boolean).
                 "quote_verbatim_grade": None, "quote_missing_tokens": None,
                 "audit_note": None, "audit_note_missing": False}
        if audit:
            for check in AUDIT_CHECKS:
                verdict = str(audit.get(check) or "").strip().lower()
                if verdict == "bad":
                    flags["audit_bad"].append(check)
                elif verdict == "unsure":
                    flags["audit_unsure"].append(check)
                elif verdict == "caveat":
                    flags["audit_caveat"].append(check)
            qvp = str(audit.get("quote_verbatim_pass"))
            flags["quote_verbatim_pass"] = qvp.lower() == "true"
            flags["quote_verbatim_grade"] = audit.get("quote_verbatim_grade")
            flags["quote_missing_tokens"] = audit.get("quote_missing_tokens")
            flags["audit_note"] = audit.get("note")
            # The review prompt requires an actionable note for every non-ok
            # check; enforcing that in valid_review would cascade a bad note
            # into a dropped row, so it is surfaced as a flag instead.
            flags["audit_note_missing"] = bool(
                (flags["audit_bad"] or flags["audit_unsure"]
                 or flags["audit_caveat"])
                and not str(audit.get("note") or "").strip())
        row["intake_flags"] = flags
        # GATE-4 (C4): export the audit's reconstructed direction onto the
        # shipped row as research-export columns, so analysis stops re-joining
        # audit.jsonl by hand.  (No pipeline stage consumes these four
        # columns; an earlier comment claimed dedup would -- it never did.)
        # Top-level columns, None when no audit row exists.
        row["audit_economic_action"] = (audit or {}).get("economic_action")
        row["audit_expected_provider"] = (audit or {}).get("expected_capital_provider")
        row["audit_expected_recipient"] = (audit or {}).get("expected_capital_recipient")
        row["audit_expected_target"] = (audit or {}).get("expected_target_firm")
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("doc_date") or "9999"), r["raw_id"]))
    obs_path = os.path.join(outdir, "observations.jsonl")
    n = write_jsonl(rows, obs_path)
    q_path = os.path.join(outdir, "intake_quarantine.jsonl")
    write_jsonl(quarantine, q_path)
    man.add_output(obs_path, n)
    man.add_output(q_path, len(quarantine))
    stats = {"observations_in": len(instances), "observations_ok": n,
             "quarantined": len(quarantine),
             "audit_duplicate_raw_ids": n_dup_audit,
             "audit_missing": sum(1 for r in rows
                                  if not r["intake_flags"]["audit_present"])}
    man.done(stats)
    return stats
