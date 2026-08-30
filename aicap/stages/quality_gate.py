#!/usr/bin/env python3
"""Stage `quality_gate` -- one deterministic gate before the Fact layer.

Joins the normalization outputs into observations_final.jsonl and stamps
observation_status:
  canonical_eligible  clean row
  review_required     flagged but flows (reasons listed)
  quarantined         unusable for Fact identity -- excluded from the Fact
                      layer.  The exact predicate: no resolved target id,
                      OR an amount_text that produced no parsed magnitude
                      (a stated amount nobody could read is not a usable
                      identity anchor).
Nothing admitted by Stage 3 is dropped silently; quarantine is a status, and
the row stays in the table.
"""
from __future__ import annotations

import os

from .common import (Manifest, coerce_num, is_nullish, mw_value,
                     read_jsonl, write_jsonl)

# Amount flags that mean "the LLM lane FAILED and a distrusted deterministic
# value was silently kept" -- these block (review_required); the escalation
# reasons that merely sent a row to the LLM remain info-tier.
_AMOUNT_FALLBACK_BLOCKING = frozenset({
    "llm_all_null", "llm_quarantine_det_fallback",
    "llm_not_reconstructible_det_fallback", "rule_not_reconstructible"})


def run(outdir: str) -> dict:
    man = Manifest(outdir, "quality_gate")
    obs = read_jsonl(os.path.join(outdir, "observations.jsonl"))
    amounts = {r["raw_id"]: r for r in read_jsonl(os.path.join(outdir, "obs_amounts.jsonl"))}
    tflags = {r["raw_id"]: r for r in read_jsonl(os.path.join(outdir, "obs_time_flags.jsonl"))}
    resolved = {r["raw_id"]: r for r in read_jsonl(os.path.join(outdir, "observations_resolved.jsonl"))}
    for name in ("observations.jsonl", "obs_amounts.jsonl", "obs_time_flags.jsonl",
                 "observations_resolved.jsonl"):
        man.add_input(os.path.join(outdir, name))

    rows, n_status = [], {"canonical_eligible": 0, "review_required": 0, "quarantined": 0}
    for o in obs:
        rid = o["raw_id"]
        a = amounts.get(rid, {})
        t = tflags.get(rid, {})
        r = resolved.get(rid, {})
        reasons = []
        target_ids = [i for i in (r.get("target_entity_ids") or []) if i]
        has_amount = any(coerce_num(a.get(k)) is not None for k in
                         ("amount_native_point", "amount_native_low", "amount_native_high"))
        has_capacity = mw_value(o) is not None
        if not target_ids:
            reasons.append("no_resolved_target")
        if not has_amount and not has_capacity:
            reasons.append("no_magnitude")
        quarantine = ("no_resolved_target" in reasons) or ("no_magnitude" in reasons and
                                                          not is_nullish(o.get("amount_text")))
        # GATE-3 (D2): review flags split into BLOCKING (true actionables --
        # status becomes review_required) and INFO (metadata -- recorded on
        # the row, never a status change).  Ends the 35%-review alarm fatigue.
        blocking, info = [], []
        if a.get("amount_method") == "failed":
            blocking.append("amount_parse_failed")
        if a.get("amount_flags"):
            info.append("amount_flags:" + ",".join(a["amount_flags"]))  # full list
        # Fallback-FAILURE flags are blocking, not info: the rules escalated
        # this row because they distrusted their own parse, and the LLM then
        # failed to replace it -- the retained value is suspect by
        # construction (a digit-guard-failing value kept after an LLM
        # abstention is the sharpest case).  Escalation-REASON flags stay
        # info: they explain why the LLM was consulted, not that it failed.
        fallback_bad = sorted(set(a.get("amount_flags") or [])
                              & _AMOUNT_FALLBACK_BLOCKING)
        if fallback_bad:
            blocking.append("amount_fallback:" + ",".join(fallback_bad))
        extracted_bound = o.get("amount_bound_type_extracted")
        normalized_bound = a.get("amount_bound_type")
        if (has_amount and extracted_bound not in (None, "", "unclear")
                and normalized_bound not in (None, "", "unclear")
                and extracted_bound != normalized_bound):
            blocking.append(
                f"amount_bound_disagreement:{extracted_bound}_vs_{normalized_bound}")
        cur = (a.get("amount_currency") or "unknown").upper()
        if has_amount and cur not in ("USD", "UNKNOWN"):
            # PI ruling 2026-08-18: a correct figure in the correct foreign
            # currency is sufficient -- USD conversion is NOT required.
            # Info-tier unconditionally (not gated on amount_usd_reported).
            info.append(f"non_usd_unconverted:{cur}")
        if o.get("amount_needs_review"):
            blocking.append("stage2_amount_review")
        # 2026-08-19 ruling: time extraction is an independent layer -- a
        # fact is NEVER blocked because its time extraction is imperfect.
        # Time signals are info-tier; the protective action is claim-level
        # (screens / escalation verdicts demote or drop claims, and the fact
        # falls back to disclosure_only in the temporal profile).
        if t.get("time_needs_review"):
            info.append("time_needs_review")
        flags = o.get("intake_flags") or {}
        # A row that never received its Stage-3 audit passed the admission
        # lane unexamined (review under-production / batch failure); it flows,
        # but never as clean canonical evidence.
        if not flags.get("audit_present", True):
            blocking.append("audit_missing")
        if flags.get("audit_note_missing"):
            info.append("audit_note_missing")
        audit_bad = list(flags.get("audit_bad") or [])
        # Same ruling for the Stage-3 time_ok audit verdict: recorded, never
        # blocking (the READ prompt already receives the stage3_time_flag
        # warning and the profile demotes the row's claims).
        if "time_ok" in audit_bad:
            audit_bad.remove("time_ok")
            info.append("audit_time_bad")
        # A direction-fixed row (two-signal consensus swap/clear, or a signed
        # human override) still carries the auditor's anchor_direction=bad --
        # that verdict graded the PRE-fix orientation and is stale by
        # construction: on admitted rows, anchor_direction=bad occurs ONLY
        # when the fix lane corrected the row (unfixed cases drop or queue).
        # Blocking on it would send every rescued row straight back to review
        # and undo the fix lane's purpose.  Discount that one check; every
        # other bad check still blocks.
        if o.get("direction_fix_applied") and "anchor_direction" in audit_bad:
            audit_bad.remove("anchor_direction")
            info.append("direction_fixed:"
                        + str(o.get("direction_fix_source") or "unknown"))
        if audit_bad:
            blocking.append("audit_bad:" + ",".join(audit_bad))
        if flags.get("audit_unsure"):
            info.append("audit_unsure:" + ",".join(flags["audit_unsure"]))
        # GATE-2: the two formerly-dark audit flags, surfaced info-tier
        if flags.get("audit_caveat"):
            info.append("audit_caveat:" + ",".join(flags["audit_caveat"][:4]))
        if o.get("audit_should_mismatch"):
            info.append("audit_should_mismatch")
        # Verbatim quote grades are INFO-TIER across the board (user decision
        # 2026-08-28, superseding GATE-1' blocking-on-fatal): the extraction
        # contract only requires that a quote IS there; how verbatim it is
        # stays measured (pass/drift/reorder/fatal, computed free in code)
        # and recorded, but never changes status.  NOTE what this trades: a
        # `fatal` grade flags a quoted token absent from the source --
        # historically ~3 per 10k rows, typically an invented numeral -- and
        # such rows now stay canonical_eligible (visible via verbatim_fatal
        # in info_flags).
        grade = flags.get("quote_verbatim_grade")
        if grade in ("drift", "reorder", "fatal"):
            info.append("verbatim_" + grade)
        elif grade is None and flags.get("quote_verbatim_pass") is False:
            info.append("verbatim_fail_legacy")
        # Contract defects recorded by the deterministic scanner (2026-08-28
        # policy: admitted + defaulted at assembly, never repaired) BLOCK:
        # a structurally defective observation flows for visibility but is
        # review_required -- excluded from clean canonical elections and
        # filterable in every export.  The paper's error rate reads straight
        # off these flags.
        if o.get("schema_defects"):
            blocking.append("schema_defects:" + str(o["schema_defects"])[:120])
        # Ungrounded evidence spans cleared at assembly: info-tier by design
        # (evidence is provenance; the field values are Stage-3-audited and
        # the quote is verbatim-gated -- clearing must never change status).
        if o.get("evidence_cleared"):
            info.append("evidence_ungrounded:" + str(o["evidence_cleared"]))
        # collective target is a property, not a problem (info); read the
        # POST-REGISTRY flag -- a superset of the raw stage-2 flag (FLD).
        if r.get("is_collective_target"):
            info.append("collective_target")

        if quarantine:
            status = "quarantined"
        elif blocking:
            status = "review_required"
        else:
            status = "canonical_eligible"
        n_status[status] += 1

        merged = dict(o)
        merged.update({k: a.get(k) for k in
                       ("amount_native_point", "amount_native_low", "amount_native_high",
                        "amount_native_mid",                       # AMT-3
                        "amount_usd_reported", "usd_reported_source",  # AMT-2
                        "amount_currency", "amount_bound_type", "amount_period_scope",
                        "amount_method", "amount_value_source", "amount_flags",
                        "amount_currency_raw", "amount_currency_method",
                        "amount_currency_evidence", "amount_currency_conflict")})
        merged.update({k: r.get(k) for k in
                       ("target_entity_ids", "target_canonicals", "is_collective_target",
                        "counterparty_entity_ids", "parent_entity_ids",
                        "other_party_entity_ids", "object_id", "object_canonical",
                        "object_type", "parent_object_id",
                        "parent_object_canonical")})
        merged["time_needs_review"] = bool(t.get("time_needs_review"))
        merged["n_time_claims"] = t.get("n_claims", 0)
        merged["observation_status"] = status
        merged["blocking_reasons"] = sorted(set(blocking))
        merged["info_flags"] = sorted(set(info))
        # union kept for export back-compat (verified: nothing downstream
        # reads it -- fact_dedup filters on observation_status only)
        merged["status_reasons"] = sorted(set(reasons + blocking + info))
        rows.append(merged)

    path = os.path.join(outdir, "observations_final.jsonl")
    n = write_jsonl(rows, path)
    man.add_output(path, n)
    stats = {"rows": n, **n_status,
             "rows_with_blocking": sum(1 for x in rows if x["blocking_reasons"]),
             "rows_info_only": sum(1 for x in rows
                                   if x["info_flags"] and not x["blocking_reasons"]
                                   and x["observation_status"] != "quarantined")}
    man.done(stats)
    return stats
