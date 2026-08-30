#!/usr/bin/env python3
"""Deterministic audit-evidence routing for Stage-3 admission.

The Stage-3 auditor reliably DETECTS problems in its structured fields and
notes but tends to under-grade them.  Its grades stay untouched
(ok|caveat|unsure|bad); instead this module turns the auditor's own explicit
evidence into deterministic routing:

  * direction_escalation(audit, row) -> reason | None
      Fires when the auditor's reconstructed roles contradict the row:
      (a) expected_target_firm != target_firm (normalized ';'-set compare), or
      (b) counterparty is empty while expected_counterparty names providers.
      Escalated rows join the FAIL-CLOSED human direction lane (excluded
      until a signed override), exactly like anchor_direction=bad rows.

  * should_mismatch(audit, row) -> "v1|v2" | None
      Fires when the audit note contains an explicit `should=<value>` whose
      value differs from every corresponding field on the row.  The row is
      ADMITTED and carries audit_should_mismatch as a review reason
      (informational, never a drop).

  * role_slot_repair(row) -> (target_firm, counterparty) | None
      NOT WIRED into the admission chain -- kept for reference only.  Would
      re-derive the slots from the row's own emitted roles without consulting
      the auditor.

  * consensus_clear_counterparty(audit, row) -> "model_consensus_clear" | None
      Fires when the extractor named no party on the other end AND the auditor
      reconstructed none either.  The named parties are moved to other_parties
      and the row is ADMITTED with counterparty=None.
      
* auto_direction_fix(audit, row) -> "model_consensus" | None
      Fires only when the auditor's expected pair AND the extractor's own
      emitted roles INDEPENDENTLY agree the slots are exactly reversed.  The
      row is swapped and ADMITTED with a signed trail in
      auto_direction_fixes.jsonl.  Human overrides still take precedence.

Both routing functions are pure (no I/O).
"""
from __future__ import annotations
import json
import re

from .config import AMOUNT_BASIS_SET, INSTRUMENT_CLASS_SET

_LEGAL_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "lp", "llp", "plc", "holdings", "holding", "group",
    "sa", "nv", "ag", "gmbh", "pte", "pty",
}
_SHOULD_RE = re.compile(r"should\s*=\s*([^;|]{1,80})", re.I)


def _norm_name(name) -> str:
    """Loose org-name key: lower, punctuation-free, legal suffixes stripped."""
    toks = re.split(r"[^\w]+", str(name or "").lower())
    return " ".join(t for t in toks if t and t not in _LEGAL_TOKENS)


_MAX_NAME_TOKENS = 10          # above this a member is auditor PROSE, not a name


def _name_set(joined) -> frozenset:
    """';'-compound -> order-free set of normalized member names.  A member
    longer than _MAX_NAME_TOKENS is auditor PROSE, not an org name ("Blue Owl
    continuously of...") -- dropped so it can neither match nor mismatch.

    The cap must stay GENEROUS.  Dropping is asymmetric: when one side of a
    comparison is dropped and the other is not, the two sets differ and every
    caller reads that as a direction conflict.  A cap of 6 did exactly that to
    long-but-legitimate names ("ACS and GIP Data Center JV" is 6 tokens after
    suffix stripping, already at the edge), manufacturing spurious
    expected_target_mismatch escalations.  Prose from the auditor runs well
    past 10 tokens, so 10 separates the two populations with room to spare."""
    out = set()
    for p in str(joined or "").split(";"):
        n = _norm_name(p)
        if n and len(n.split()) <= _MAX_NAME_TOKENS:
            out.add(n)
    return frozenset(out)


def direction_escalation(audit, row):
    """Deterministic direction-lane trigger from the auditor's own structured
    role reconstruction.  Returns a reason string or None.  Never fires when
    the auditor produced no expectation (expected_* null)."""
    if not isinstance(audit, dict):
        return None
    exp_target = _name_set(audit.get("expected_target_firm"))
    if exp_target and exp_target != _name_set(row.get("target_firm")):
        return "expected_target_mismatch"
    exp_cp = _name_set(audit.get("expected_counterparty"))
    if exp_cp and not _name_set(row.get("counterparty")):
        return "counterparty_omitted"
    return None

def _parent_set(row) -> frozenset:
    """Normalized name-set of the row's own target_parent (plain string,
    ';'-compound, or JSON list).  Used by the granularity exemption below."""
    tp = row.get("target_parent")
    if tp is None:
        return frozenset()
    if isinstance(tp, (list, tuple)):
        parts = [str(x) for x in tp]
    else:
        s = str(tp).strip()
        if s.startswith("["):
            try:
                parts = [str(x) for x in json.loads(s)]
            except (ValueError, TypeError):
                parts = [s]
        else:
            parts = [s]
    out = set()
    for p in parts:
        for q in str(p).split(";"):
            n = _norm_name(q)
            if n and len(n.split()) <= _MAX_NAME_TOKENS:
                out.add(n)
    return frozenset(out)


def emitted_role_mismatch(row):
    """Extractor self-consistency: slots must equal emitted roles as a set,
    oriented by economic_action. Returns a reason string or None.

    GRANULARITY EXEMPTION (2026-08-19): a target-side difference where the
    emitted role is exactly the row's OWN declared target_parent is entity
    granularity, not direction.  ENTITY GRANULARITY requires the slot to hold
    the most specific vehicle (the SPV / project company) while the economic
    role naturally names the sponsor that acts through it -- the same party
    at two granularities, one flow, no reversal.  General by construction:
    it keys on the row's own parent declaration, never on any entity list."""
    act = (row.get("economic_action") or "unclear").lower()
    prov = _name_set(row.get("capital_provider"))
    recp = _name_set(row.get("capital_recipient"))
    tgt  = _name_set(row.get("target_firm"))
    cp   = _name_set(row.get("counterparty"))
    parents = _parent_set(row)

    if act in ("financing",) and prov and recp:
        if tgt and tgt != recp and not (recp and recp <= parents):
            return "emitted_role_mismatch"
        if cp and cp != prov:
            return "emitted_role_mismatch"

    if act in ("spending", "acquisition", "guarantee") and prov and recp:
        if tgt and tgt != prov and not (prov and prov <= parents):
            return "emitted_role_mismatch"
        if cp and cp != recp:
            return "emitted_role_mismatch"

    return None


def inverted_anchor(row):
    """The half-null corner emitted_role_mismatch cannot see: only ONE role is
    emitted, and it sits on the WRONG side of the slot map.

      financing:  target == capital_provider while capital_recipient is null
                  (the sponsors were anchored; the unnamed recipient vanished)
      spending / acquisition / guarantee:
                  target == capital_recipient while capital_provider is null

    Under financing the target must be the RECIPIENT, so a row whose target
    IS the sole named provider is inverted by its own testimony -- no auditor
    needed.  This is the class that shipped unflagged (2026-08-18 review):
    emitted_role_mismatch requires BOTH roles, the auditor rationalized the
    anchoring, and consensus_clear needs a counterparty.  Fail-closed."""
    act = (row.get("economic_action") or "unclear").lower()
    tgt = _name_set(row.get("target_firm"))
    if not tgt:
        return None
    prov = _name_set(row.get("capital_provider"))
    recp = _name_set(row.get("capital_recipient"))
    cp = _name_set(row.get("counterparty"))
    slots = tgt | cp
    # The general form: under financing SOME slot must hold the recipient.
    # If EVERY filled slot party is a declared provider and no recipient is
    # emitted, the flow's receiving end is missing entirely -- whether the
    # providers were anchored jointly (tgt == prov) or split across the two
    # slots (tgt|cp == prov).  Symmetric for the paying end under spending.
    if act == "financing" and prov and not recp and slots <= prov:
        return "provider_anchored_financing"
    if act in ("spending", "acquisition", "guarantee") \
            and recp and not prov and slots <= recp:
        return "recipient_anchored_spending"
    return None


def degenerate_slots(row):
    """target_firm and counterparty are defined as the two ENDS of one flow;
    the SAME name-set in both slots is self-contradiction, not a judgment
    call.  DIR-4 deliberately never fires on it (its guard needs tgt != cp),
    so without this check the row falls through every lane."""
    tgt, cp = _name_set(row.get("target_firm")), _name_set(row.get("counterparty"))
    return "target_equals_counterparty" if (tgt and cp and tgt == cp) else None


def degenerate_slot_fix(audit, row):
    """Two-signal repair for the degenerate case, mirroring auto_direction_fix:

      A. the extractor's OWN emitted roles imply a coherent pair whose target
         matches the current slot and whose counterparty is a DIFFERENT,
         non-empty party;
      B. the auditor's reconstructed pair says exactly the same.

    Only the duplicated counterparty slot is rewritten, with the extractor's
    own surface form; the target never moves.  Returns the corrected
    counterparty surface string, or None."""
    if not isinstance(audit, dict):
        return None
    if not degenerate_slots(row):
        return None
    imp_t, imp_c = implied_roles(row, audit)
    tgt = _name_set(row.get("target_firm"))
    if _name_set(imp_t) != tgt:
        return None
    ic = _name_set(imp_c)
    if not ic or ic == tgt:
        return None
    if _name_set(audit.get("expected_target_firm")) != tgt:      # signal B
        return None
    if _name_set(audit.get("expected_counterparty")) != ic:
        return None
    return imp_c

def role_slot_repair(row):
    """NOT WIRED (2026-08-12 decision): the swap lane deliberately stays on
    auto_direction_fix alone.  Kept because the analysis behind it is worth
    preserving and re-wiring is a two-line change in extraction.py -- see
    docs/PIPELINE_REVISION_ADDENDUM_A_2026-08-12.md.  Do not call it without
    re-reading that note.

    DIR-3b: the row's own emitted roles are coherent, but its slots
    contradict them -- re-derive the slots FROM the roles.

    Why the roles win.  The extract prompt runs role resolution in two moves:
    step 2 identifies CAPITAL PROVIDER and CAPITAL RECIPIENT independently (the
    reasoning), step 3 maps them onto target_firm / counterparty (a mechanical
    rewrite).  Grammatical-subject anchoring corrupts step 3, not step 2: on
    every observed mismatch the emitted roles were right and only the slots were
    wrong ("Amazon.com pumped $4bn into Anthropic" -> provider=Amazon,
    recipient=Anthropic, but target=Amazon).  This repair therefore needs no
    second opinion: the row already contains its own correction.

    Deliberately NOT gated on the auditor.  The same audit prompt and model gave
    opposite anchor_direction verdicts on byte-identical rows across runs, so
    any rescue that requires the auditor to object is luck.

    Fires ONLY on an unambiguous SIDE SWAP -- the current target is exactly the
    party that belongs on the OTHER side.  A slot and a role may legitimately
    differ in FORM without differing in substance: target_firm carries rules the
    roles do not (ENTITY GRANULARITY -- the contracting subsidiary/SPV over its
    parent, copied as written, never "corrected" to a better-known affiliate).
    Rewriting the slot from the role on a mere wording or granularity difference
    would overwrite the more specific entity with a coarser one, so those rows
    stay in the human lane.

    Returns (target_firm, counterparty) as raw surface strings, or None."""
    if not emitted_role_mismatch(row):
        return None
    tgt, cp = implied_roles(row)
    if not tgt or not cp:
        return None
    cur = _name_set(row.get("target_firm"))
    if not cur or cur == _name_set(tgt):
        return None                      # already right, or nothing to compare
    if cur != _name_set(cp):
        return None                      # not a clean side swap -- leave to a human
    return tgt, cp


def implied_roles(row, audit=None):
    """(target_firm, counterparty) as implied by the EXTRACTOR's own emitted
    roles, returned as RAW surface strings -- or (None, None) when the roles
    are absent or the action is unusable.  Orientation prefers the row's own
    economic_action; the auditor's is only a fallback."""
    act = (row.get("economic_action")
           or (audit or {}).get("economic_action") or "unclear").lower()
    prov, recp = row.get("capital_provider"), row.get("capital_recipient")
    if not prov or not recp:
        return None, None
    if act == "financing":
        return recp, prov
    if act in ("spending", "acquisition", "guarantee"):
        return prov, recp
    return None, None


def auto_direction_fix(audit, row):
    """DIR-4: TWO INDEPENDENT signals must agree that target/counterparty are
    exactly reversed before Stage 3 swaps them without a human.

      A. the auditor's reconstructed pair IS the row's pair, swapped
         (expected_target == counterparty AND expected_counterparty == target);
      B. the extractor's OWN emitted roles imply the same swap.

    Both are required because either alone is unreliable: the auditor
    sometimes rationalizes whatever the slots already say, and the emitted
    roles can misread "provider" as the supplier of goods.  Agreement between
    two differently-derived signals is the evidence.  Returns
    "model_consensus" or None.  Never fires on empty or identical slots, and
    never on partial/multi-party asymmetry -- those keep going to the human."""
    if not isinstance(audit, dict):
        return None
    tgt, cp = _name_set(row.get("target_firm")), _name_set(row.get("counterparty"))
    if not tgt or not cp or tgt == cp:
        return None
    exp_t = _name_set(audit.get("expected_target_firm"))
    exp_c = _name_set(audit.get("expected_counterparty"))
    if not (exp_t and exp_c and exp_t == cp and exp_c == tgt):   # signal A
        return None
    imp_t, imp_c = implied_roles(row, audit)                      # signal B
    if _name_set(imp_t) != cp or _name_set(imp_c) != tgt:
        return None
    return "model_consensus"


def consensus_clear_counterparty(audit, row):
    """Mirror of auto_direction_fix: TWO INDEPENDENT signals agree the row's
    counterparty is not a capital endpoint at all (bookrunners, arrangers,
    advisers named in a clause about underwriting rather than about who pays).

      A. the extractor itself named NO party on the other end of this flow
         (capital_provider null for financing, capital_recipient null for
         spending/acquisition/guarantee);
      B. the auditor reconstructed no counterparty either.

    The parties are MOVED to other_parties by the caller, never deleted -- they
    are real participants, just not the other end of THIS flow.  Returns
    "model_consensus_clear" or None."""
    if not isinstance(audit, dict):
        return None
    if not _name_set(row.get("counterparty")):
        return None
    act = (row.get("economic_action") or audit.get("economic_action") or "").lower()
    if act == "financing":
        emitted_none = not _name_set(row.get("capital_provider"))
    elif act in ("spending", "acquisition", "guarantee"):
        emitted_none = not _name_set(row.get("capital_recipient"))
    else:
        return None
    audit_none = not _name_set(audit.get("expected_counterparty"))
    return "model_consensus_clear" if (emitted_none and audit_none) else None


# A should= proposal is only actionable when its value belongs to a KNOWN
# controlled vocabulary -- then it is compared against exactly the row
# field(s) that vocabulary governs.  Free-text proposals (names, amounts,
# "null", prose) are SKIPPED: direction proposals are handled by
# direction_escalation, amount doubts by the amount review flags.
_COMMITMENT_VOCAB = {"terminated", "cancelled", "delayed_or_paused", "operational",
                     "under_construction", "contracted", "approved", "announced",
                     "planned", "proposed", "unclear"}
_DEAL_VOCAB = {"amendment", "update", "new_commitment", "periodic_disclosure",
               "current_snapshot", "repeat_reference", "unclear"}
_SCOPE_VOCAB = {"entire_amount", "mostly_infra", "partial_unallocated", "unclear"}
_GEO_VOCAB = {"us_only", "includes_non_us", "unknown"}
# Fallback values ARE vocabulary members, but proposing one against a specific
# row value is a hedge, not an alternative classification -- "should=other"
# mismatches every specific value by construction.  Skipped entirely.
_NONINFORMATIVE_SHOULD = {"other", "unknown", "unclear"}

# vocabulary -> the row fields it governs (a value in several vocabs flags only
# when it differs from EVERY governed field -- conservative).
_VOCAB_FIELDS = (
    (AMOUNT_BASIS_SET, ("amount_basis",)),
    (INSTRUMENT_CLASS_SET, ("instrument_class",)),
    (_COMMITMENT_VOCAB, ("commitment_status",)),
    (_DEAL_VOCAB, ("deal_status",)),
    (_SCOPE_VOCAB, ("ai_infra_scope",)),
    (_GEO_VOCAB, ("geo_scope",)),
)


def should_mismatch(audit, row):
    """Explicit `should=<enum value>` proposals in the audit note that differ
    from the row's value for every field governed by that vocabulary.
    '(acceptable)' phrasing does NOT suppress the flag -- an explicitly
    different value always surfaces (admit + review reason)."""
    if not isinstance(audit, dict):
        return None
    note = str(audit.get("note") or "")
    if not note:
        return None
    hits = []
    for m in _SHOULD_RE.finditer(note):
        value = re.sub(r"\(.*?\)", "", m.group(1)).strip(" .,\"'")
        # enum vocab values are single snake_case tokens; the capture may trail
        # into prose ("should=under_construction per quote") -- first token only
        toks = value.lower().split()
        key = toks[0].strip(".,;:'\"") if toks else ""
        if key in _NONINFORMATIVE_SHOULD:
            continue                          # hedge, not a proposal
        fields = [f for vocab, fs in _VOCAB_FIELDS if key in vocab for f in fs]
        if not fields:
            continue                          # not a controlled-vocabulary proposal
        if all(key != str(row.get(f) or "").lower().strip() for f in fields):
            hits.append(key)
    return "|".join(dict.fromkeys(hits)) or None
