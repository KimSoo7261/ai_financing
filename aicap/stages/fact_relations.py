#!/usr/bin/env python3
"""Stage ``build_fact_relations`` -- typed Fact-to-Fact relation graph.

Atomic deduplication answers whether two observations belong to one Fact.
This stage runs only after Fact identity, elected state, and temporal profiles
are stable.  It answers the different question required by additive work:
whether two *different* Facts may be added without overlap.

The stage remains Fact-native: it creates no Deal object.  Deterministic
multi-key blocking constructs a high-recall set of potentially related Fact
pairs, high-precision rules settle explicit guidance revisions, and one
bounded LLM request compares a pivot with up to ``MAX_CANDIDATES_PER_PROMPT``
candidates.  Uncertainty always blocks joint addition; only an explicit,
high-confidence ``independent`` verdict releases a risky pair.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from .common import (Manifest, SOURCE_RANK, read_jsonl, sha16,
                     write_jsonl)
from .judge import Judge, content_id


MAX_CANDIDATES_PER_PROMPT = 8
MAX_BLOCK_SIZE = 80
AMOUNT_CONTAINMENT_TOL = 0.02
EVIDENCE_CHARS = 700

MONEY_RELATION_INSTRUMENTS = frozenset({
    "corporate_debt", "project_debt", "lease", "supplier_purchase",
    "compute_capacity_contract", "guarantee", "ppa_power", "jv_spv",
    "equity_contribution", "mixed_financing", "grant_subsidy", "m_and_a",
    "asset_acquisition", "general_capex",
})

RAW_RELATIONS = frozenset({
    "pivot_contains_candidate", "candidate_contains_pivot",
    "partial_overlap", "revision_of", "alternative_measure_of",
    "independent", "same_atomic_fact", "unclear",
})
RELATIONS = frozenset({
    "contains", "partial_overlap", "revision_of",
    "alternative_measure_of", "independent", "same_atomic_fact", "unclear",
})
CONFIDENCES = frozenset({"high", "medium", "low"})

AGGREGATE_RE = re.compile(
    r"\b(total|aggregate|aggregated|including|includes|included|of which|"
    r"consists of|comprised of|outstanding|overall)\b", re.I)
SEPARATE_RE = re.compile(
    r"\b(in addition|additional|another|separate|second|distinct)\b", re.I)
REVISION_RE = re.compile(
    r"\b(updated?|revised?|increased?|decreased?|amended?|now|formerly|"
    r"previously)\b", re.I)


RELATION_SYSTEM_PROMPT = r"""You are the regv1 FACT RELATION JUDGE for a conservative capital-obligation registry.

The facts shown have already passed atomic Fact deduplication. Do not extract new facts, correct amounts, or construct deals. Compare the PIVOT FACT with every CANDIDATE FACT and classify the relationship between their economic quantities.

Allowed relations:
- pivot_contains_candidate: the candidate's entire measured obligation is included in the pivot's quantity, using compatible measures, obligation states, time snapshots, and scopes.
- candidate_contains_pivot: the reverse direction.
- partial_overlap: the facts cover some of the same obligation, facility, contract, project component, or reporting perimeter, but neither is proven to contain the other fully.
- revision_of: successive estimates, amendments, snapshots, replacements, or updates of the same economic quantity; they must not be added.
- alternative_measure_of: the same underlying obligation under different measures, such as recognized lease liability versus undiscounted future payments, outstanding principal versus authorized capacity, or annual payment versus lifetime value.
- independent: the evidence affirmatively establishes separate obligations or disjoint components that may coexist.
- same_atomic_fact: duplicate tellings of one atomic fact escaped the earlier deduplication pass.
- unclear: the supplied evidence does not establish the relationship.

Rules:
1. A larger amount alone NEVER proves containment.
2. A shared company, project, counterparty, amount, or date alone NEVER proves containment or independence.
3. Total/aggregate/including/of-which language proves containment only when the evidence connects the displayed quantities or components.
4. A binding contingent obligation is not speculative merely because payout depends on a future condition.
5. Authorized capacity, committed-undrawn capacity, funded principal, recognized liability, future payments, annual flows, lifetime values, and maximum contingent exposure are different measures unless the evidence establishes otherwise.
6. Read source metadata as supplied. Use only the structured fields and quoted evidence. Never invent a number, participant, or contract.
7. Independence requires affirmative evidence. When it cannot be established, answer unclear.
8. Return exactly one decision for every CANDIDATE_ID and do not return any other candidate id.

Reply JSON ONLY:
{"pivot_id":"...","decisions":[{"candidate_id":"...","relation":"pivot_contains_candidate|candidate_contains_pivot|partial_overlap|revision_of|alternative_measure_of|independent|same_atomic_fact|unclear","confidence":"high|medium|low","evidence_ids":["P-A","C1-A"],"reason":"short evidence-based reason"}]}"""


def _clean_text(value, limit: int = EVIDENCE_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("."), cut.rfind(";"), cut.rfind(","))
    return (cut[:boundary] if boundary >= limit // 2 else cut).rstrip() + "..."


def _as_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _amount_value(amount: dict):
    for key in ("point", "high", "low"):
        value = _as_float((amount or {}).get(key))
        if value is not None:
            return value
    return None


def _amount_interval(amount: dict):
    amount = amount or {}
    point = _as_float(amount.get("point"))
    low = _as_float(amount.get("low"))
    high = _as_float(amount.get("high"))
    if low is None:
        low = point if point is not None else high
    if high is None:
        high = point if point is not None else low
    if low is None or high is None:
        return None
    return (min(low, high), max(low, high))


def _measure_group(instrument: str) -> str:
    if instrument in {"corporate_debt", "project_debt"}:
        return "debt"
    return instrument or "unknown"


def _measure_family(instrument: str, basis: str) -> str:
    group = _measure_group(instrument)
    basis = basis or "unknown"
    if group == "debt":
        return {
            "debt_principal": "debt_principal",
            "recognized_balance": "debt_recognized_balance",
            "carrying_value": "debt_recognized_balance",
            "authorized_amount": "debt_authorized_capacity",
            "undiscounted_commitment": "debt_future_commitment",
            "facility_commitment": "debt_authorized_capacity",
            "financing_total": "debt_financing_total",
        }.get(basis, "debt_unknown")
    if group == "lease":
        return {
            "recognized_balance": "lease_recognized_liability",
            "carrying_value": "lease_recognized_liability",
            "undiscounted_commitment": "lease_future_payments",
            "contract_value": "lease_contract_value",
            "authorized_amount": "lease_authorized_capacity",
        }.get(basis, "lease_unknown")
    if group == "supplier_purchase":
        return {
            "recognized_balance": "purchase_recognized_balance",
            "carrying_value": "purchase_recognized_balance",
            "undiscounted_commitment": "purchase_lifetime_commitment",
            "contract_value": "purchase_contract_value",
            "deposit_or_prepayment": "purchase_prepayment",
            "authorized_amount": "purchase_authorized_capacity",
        }.get(basis, "purchase_unknown")
    if group == "guarantee":
        return "guarantee_max_exposure" if basis == "max_exposure" \
            else "guarantee_unknown"
    return f"{group}_{basis}"


def _obligation_state(state: dict) -> str:
    status = str(state.get("commitment_status_latest") or "").casefold()
    if status in {"funded", "drawn", "issued", "closed", "completed"}:
        return "funded_or_drawn"
    if status in {"committed", "contracted", "binding", "signed", "executed"}:
        return "binding_commitment"
    if status in {"authorized", "approved"}:
        return "authorized_only"
    if status in {"proposed", "discussed", "planned_nonbinding", "nonbinding"}:
        return "proposed"
    if state.get("canonical_instrument") == "guarantee":
        return "contingent_binding_or_unclear"
    return status or "unclear"


def _set(values) -> frozenset:
    return frozenset(value for value in (values or []) if value)


@dataclass(frozen=True)
class FactNode:
    fact_id: str
    raw_ids: tuple[str, ...]
    canonical_id: str
    amount_id: str
    target_ids: frozenset[str]
    parent_ids: frozenset[str]
    counterparty_ids: frozenset[str]
    object_ids: frozenset[str]
    target_names: tuple[str, ...]
    parent_names: tuple[str, ...]
    counterparty_names: tuple[str, ...]
    object_names: tuple[str, ...]
    aggregation_level: str
    instrument: str
    basis: str
    measure_group: str
    measure_family: str
    obligation_state: str
    amount: float | None
    amount_interval: tuple[float, float] | None
    currency: str
    first_disclosed_at: str
    valid_from: str
    valid_to: str
    series_id: str
    aggregate_cue: bool
    separate_cue: bool
    revision_cue: bool

    @property
    def closure(self) -> frozenset[str]:
        return self.target_ids | self.parent_ids


def _build_nodes(facts: list[dict], states: list[dict], profiles: list[dict],
                 obs_by_id: dict[str, dict],
                 entity_names: dict[str, str] | None = None
                 ) -> dict[str, FactNode]:
    entity_names = entity_names or {}
    state_by = {row["fact_id"]: row for row in states}
    profile_by = {row["fact_id"]: row for row in profiles}
    nodes = {}
    for fact in facts:
        fact_id = fact["fact_id"]
        state = state_by.get(fact_id)
        if state is None:
            continue
        instrument = str(state.get("canonical_instrument") or "unknown")
        basis = str(state.get("canonical_basis") or "unknown")
        amount = state.get("canonical_amount") or {}
        canonical_id = str(fact.get("canonical_observation_id") or "")
        amount_id = str(state.get("canonical_amount_observation_id") or canonical_id)
        # Cue routing scans every member so a source-ranked canonical that
        # omits "total" cannot hide an aggregate cue preserved in another
        # telling.  The LLM card remains bounded to at most three best rows.
        evidence_text = " ".join(
            str(obs_by_id.get(raw_id, {}).get("verbatim_quote") or "")
            for raw_id in (fact.get("raw_ids") or []) if raw_id)
        profile = profile_by.get(fact_id) or {}
        nodes[fact_id] = FactNode(
            fact_id=fact_id,
            raw_ids=tuple(sorted(fact.get("raw_ids") or [])),
            canonical_id=canonical_id,
            amount_id=amount_id,
            target_ids=_set(state.get("target_entity_ids")),
            parent_ids=_set(state.get("parent_entity_ids")),
            counterparty_ids=_set(state.get("counterparty_entity_ids")),
            object_ids=_set(state.get("object_ids")),
            target_names=tuple(sorted(str(value) for value in
                                      (state.get("target_canonicals") or []) if value)),
            parent_names=tuple(sorted(entity_names.get(value, value) for value in
                                      (state.get("parent_entity_ids") or []) if value)),
            counterparty_names=tuple(sorted(entity_names.get(value, value) for value in
                                            (state.get("counterparty_entity_ids") or [])
                                            if value)),
            object_names=tuple(sorted(str(value) for value in
                                      (state.get("object_canonicals") or []) if value)),
            aggregation_level=str(
                state.get("canonical_aggregation_level") or "unknown"),
            instrument=instrument,
            basis=basis,
            measure_group=_measure_group(instrument),
            measure_family=_measure_family(instrument, basis),
            obligation_state=_obligation_state(state),
            amount=_amount_value(amount),
            amount_interval=_amount_interval(amount),
            currency=str(amount.get("currency") or "UNKNOWN").upper(),
            first_disclosed_at=str(profile.get("first_disclosed_at") or ""),
            valid_from=str(profile.get("valid_from_best") or ""),
            valid_to=str(profile.get("valid_to_best") or ""),
            series_id=str(fact.get("series_id") or ""),
            aggregate_cue=bool(AGGREGATE_RE.search(evidence_text)),
            separate_cue=bool(SEPARATE_RE.search(evidence_text)),
            revision_cue=bool(REVISION_RE.search(evidence_text)),
        )
    return nodes


def _periods_proven_disjoint(a: FactNode, b: FactNode) -> bool:
    if not (a.valid_from and a.valid_to and b.valid_from and b.valid_to):
        return False
    return a.valid_to < b.valid_from or b.valid_to < a.valid_from


def _pair_score(a: FactNode, b: FactNode) -> int | None:
    shared_object = bool(a.object_ids & b.object_ids)
    closure = bool(a.closure & b.closure)
    counterparties = bool(a.counterparty_ids & b.counterparty_ids)
    if not shared_object and not closure:
        return None
    if a.measure_group != b.measure_group:
        return None
    if _periods_proven_disjoint(a, b) and not (a.revision_cue or b.revision_cue):
        return None

    score = 8 * int(shared_object) + 5 * int(closure) + 3 * int(counterparties)
    score += 3 * int(a.measure_family == b.measure_family)
    score += 3 * int(a.aggregate_cue or b.aggregate_cue)
    score += 2 * int(a.separate_cue or b.separate_cue)
    score += 2 * int(a.revision_cue or b.revision_cue)
    if a.amount is not None and b.amount is not None:
        ratio = min(a.amount, b.amount) / max(a.amount, b.amount)
        score += 2 * int(ratio >= 0.85)

    # A generic same-target cell is intentionally not enough when neither
    # side carries any relation signal. A shared object/counterparty, a
    # cross-measure view, or explicit aggregate/separate/revision language is
    # required to spend an LLM decision.
    signalled = (shared_object or counterparties
                 or a.measure_family != b.measure_family
                 or a.aggregate_cue or b.aggregate_cue
                 or a.separate_cue or b.separate_cue
                 or a.revision_cue or b.revision_cue)
    return score if signalled else None


def _candidate_pairs(nodes: dict[str, FactNode]) -> dict[tuple[str, str], int]:
    blocks: dict[tuple, set[str]] = defaultdict(set)
    for node in nodes.values():
        if node.instrument not in MONEY_RELATION_INSTRUMENTS or node.amount is None:
            continue
        for object_id in node.object_ids:
            blocks[("object", object_id, node.measure_group)].add(node.fact_id)
        for entity in node.closure:
            blocks[("entity", entity, node.measure_group)].add(node.fact_id)
        if node.target_ids and node.counterparty_ids:
            blocks[("parties", tuple(sorted(node.target_ids)),
                    tuple(sorted(node.counterparty_ids)),
                    node.measure_group)].add(node.fact_id)

    scored = {}
    for key, fact_ids in blocks.items():
        ordered = sorted(fact_ids)
        # An abnormally broad registry entity can otherwise manufacture an
        # accidental quadratic block.  Project/party blocks remain available;
        # the oversized entity block is surfaced in reviews by run().
        if key[0] == "entity" and len(ordered) > MAX_BLOCK_SIZE:
            continue
        for left, right in combinations(ordered, 2):
            score = _pair_score(nodes[left], nodes[right])
            if score is not None:
                pair = (left, right)
                scored[pair] = max(score, scored.get(pair, -1))
    return scored


def _evidence_rows(node: FactNode, obs_by_id: dict[str, dict]) -> list[dict]:
    selected = []
    seen = set()

    def add(raw_id):
        if raw_id and raw_id not in seen and raw_id in obs_by_id:
            seen.add(raw_id)
            selected.append(obs_by_id[raw_id])

    def score(raw_id):
        row = obs_by_id.get(raw_id, {})
        quote = str(row.get("verbatim_quote") or "")
        flags = row.get("intake_flags") or {}
        return (SOURCE_RANK.get(row.get("source_type"), 0),
                -len(flags.get("audit_bad", [])),
                int(bool(AGGREGATE_RE.search(quote)
                         or SEPARATE_RE.search(quote)
                         or REVISION_RE.search(quote))),
                len(quote), str(raw_id))

    add(node.amount_id)
    if node.raw_ids:
        add(max(node.raw_ids, key=score))
        cue_ids = [raw_id for raw_id in node.raw_ids
                   if (AGGREGATE_RE.search(str(obs_by_id.get(raw_id, {}).get(
                           "verbatim_quote") or ""))
                       or SEPARATE_RE.search(str(obs_by_id.get(raw_id, {}).get(
                           "verbatim_quote") or ""))
                       or REVISION_RE.search(str(obs_by_id.get(raw_id, {}).get(
                           "verbatim_quote") or "")))]
        if cue_ids:
            add(max(cue_ids, key=score))
    add(node.canonical_id)
    return selected[:3]


def _card(node: FactNode, obs_by_id: dict[str, dict], prefix: str):
    labels = {}
    evidence = []
    for pos, row in enumerate(_evidence_rows(node, obs_by_id)):
        label = f"{prefix}-{chr(ord('A') + pos)}"
        labels[label] = row.get("raw_id")
        evidence.extend([
            f"[{label}] source={row.get('source_type')} date={row.get('doc_date')}",
            f'quote="{_clean_text(row.get("verbatim_quote"))}"',
        ])
    interval = node.amount_interval
    amount_text = (f"{node.currency} {node.amount:g}"
                   if node.amount is not None else "none")
    if interval and interval[0] != interval[1]:
        amount_text += f" interval=[{interval[0]:g},{interval[1]:g}]"
    lines = [
        f"FACT_ID={node.fact_id}",
        f"target={';'.join(node.target_names) or '-'} "
        f"ids={','.join(sorted(node.target_ids)) or '-'}",
        f"parent={';'.join(node.parent_names) or '-'} "
        f"ids={','.join(sorted(node.parent_ids)) or '-'}",
        f"counterparty={';'.join(node.counterparty_names) or '-'} "
        f"ids={','.join(sorted(node.counterparty_ids)) or '-'}",
        f"object={';'.join(node.object_names) or '-'} "
        f"ids={','.join(sorted(node.object_ids)) or '-'}",
        f"instrument={node.instrument} basis={node.basis} "
        f"aggregation_level={node.aggregation_level}",
        f"measure_family={node.measure_family} obligation_state={node.obligation_state}",
        f"amount={amount_text}",
        f"first_disclosed={node.first_disclosed_at or '-'} "
        f"valid={node.valid_from or '-'}..{node.valid_to or '-'}",
        "EVIDENCE:",
        *evidence,
    ]
    return "\n".join(lines), labels


def _orient(a: FactNode, b: FactNode) -> tuple[FactNode, FactNode]:
    if a.aggregate_cue != b.aggregate_cue:
        return (a, b) if a.aggregate_cue else (b, a)
    if a.revision_cue != b.revision_cue:
        return (a, b) if a.revision_cue else (b, a)
    if a.amount is not None and b.amount is not None and a.amount != b.amount:
        return (a, b) if a.amount > b.amount else (b, a)
    return (a, b) if a.fact_id < b.fact_id else (b, a)


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _validator(pivot_id: str, candidate_ids: set[str],
               evidence_by_candidate: dict[str, set[str]]) -> Callable[[object], bool]:
    def valid(obj) -> bool:
        if not isinstance(obj, dict) or obj.get("pivot_id") != pivot_id:
            return False
        decisions = obj.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(candidate_ids):
            return False
        returned = []
        for decision in decisions:
            if not isinstance(decision, dict):
                return False
            candidate_id = decision.get("candidate_id")
            returned.append(candidate_id)
            if decision.get("relation") not in RAW_RELATIONS:
                return False
            if decision.get("confidence") not in CONFIDENCES:
                return False
            ids = decision.get("evidence_ids")
            allowed = evidence_by_candidate.get(str(candidate_id), set())
            if not isinstance(ids, list) or any(i not in allowed for i in ids):
                return False
            if decision.get("relation") != "unclear" and not ids:
                return False
            if not isinstance(decision.get("reason"), str) \
                    or not decision.get("reason", "").strip():
                return False
        return len(returned) == len(set(returned)) and set(returned) == candidate_ids
    return valid


def _normalize_decision(pivot: FactNode, candidate: FactNode, decision: dict,
                        label_to_raw: dict[str, str]) -> dict:
    raw_relation = decision.get("relation", "unclear")
    confidence = decision.get("confidence", "low")
    relation = raw_relation
    parent_id = child_id = newer_id = older_id = None
    if raw_relation == "pivot_contains_candidate":
        relation = "contains"
        parent_id, child_id = pivot.fact_id, candidate.fact_id
    elif raw_relation == "candidate_contains_pivot":
        relation = "contains"
        parent_id, child_id = candidate.fact_id, pivot.fact_id
    elif raw_relation == "revision_of":
        if (pivot.first_disclosed_at, pivot.fact_id) >= \
                (candidate.first_disclosed_at, candidate.fact_id):
            newer_id, older_id = pivot.fact_id, candidate.fact_id
        else:
            newer_id, older_id = candidate.fact_id, pivot.fact_id

    # Only high-confidence affirmative independence may release two risky
    # facts.  Low-confidence positive relations are still useful as a reason
    # to block, but are represented uniformly as unresolved.
    actionable = confidence in {"high", "medium"}
    if relation == "independent":
        actionable = confidence == "high"
    if not actionable or relation not in RELATIONS:
        relation = "unclear"
        parent_id = child_id = newer_id = older_id = None

    ids = sorted({label_to_raw[i] for i in decision.get("evidence_ids", [])
                  if i in label_to_raw and label_to_raw[i]})
    return {
        "fact_a_id": min(pivot.fact_id, candidate.fact_id),
        "fact_b_id": max(pivot.fact_id, candidate.fact_id),
        "relation": relation,
        "raw_relation": raw_relation,
        "parent_fact_id": parent_id,
        "child_fact_id": child_id,
        "newer_fact_id": newer_id,
        "older_fact_id": older_id,
        "confidence": confidence,
        "decision_source": "llm",
        "evidence_raw_ids": ids,
        "reason": _clean_text(decision.get("reason"), 400),
    }


def _amount_check(edge: dict, nodes: dict[str, FactNode]) -> str:
    if edge["relation"] != "contains":
        return "not_applicable"
    parent = nodes[edge["parent_fact_id"]]
    child = nodes[edge["child_fact_id"]]
    if parent.measure_family != child.measure_family:
        return "failed_incompatible_measure"
    if (parent.currency in {"", "UNKNOWN"} or child.currency in {"", "UNKNOWN"}
            or parent.currency != child.currency
            or not parent.amount_interval or not child.amount_interval):
        return "not_comparable"
    if parent.amount_interval[1] < child.amount_interval[0] * (1 - AMOUNT_CONTAINMENT_TOL):
        return "failed_parent_below_child"
    return "passed"


def _cycle_nodes(edges: list[dict]) -> set[str]:
    graph = defaultdict(set)
    indegree = defaultdict(int)
    all_nodes = set()
    for edge in edges:
        if edge.get("relation") != "contains":
            continue
        parent, child = edge.get("parent_fact_id"), edge.get("child_fact_id")
        if not parent or not child or child in graph[parent]:
            continue
        graph[parent].add(child)
        indegree[child] += 1
        indegree.setdefault(parent, 0)
        all_nodes.update((parent, child))
    queue = deque(sorted(node for node in all_nodes if indegree[node] == 0))
    visited = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for child in sorted(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return all_nodes - visited


class _UF:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _finish_edges(edges: list[dict], nodes: dict[str, FactNode],
                  reviews: list[dict]) -> list[dict]:
    for edge in edges:
        edge["amount_check"] = _amount_check(edge, nodes)
        if edge["amount_check"] in {
                "failed_parent_below_child", "failed_incompatible_measure"}:
            reviews.append({
                "reason": ("containment_measure_contradiction"
                           if edge["amount_check"] == "failed_incompatible_measure"
                           else "containment_amount_contradiction"),
                "fact_ids": [edge["fact_a_id"], edge["fact_b_id"]],
                "detail": edge.get("reason", ""),
            })
            edge["raw_relation"] = edge["relation"]
            edge["relation"] = (
                "alternative_measure_of"
                if edge["amount_check"] == "failed_incompatible_measure"
                else "unclear")
            edge["parent_fact_id"] = edge["child_fact_id"] = None

    cycle = _cycle_nodes(edges)
    if cycle:
        for edge in edges:
            if (edge.get("relation") == "contains"
                    and edge.get("parent_fact_id") in cycle
                    and edge.get("child_fact_id") in cycle):
                reviews.append({
                    "reason": "containment_cycle",
                    "fact_ids": [edge["fact_a_id"], edge["fact_b_id"]],
                    "detail": edge.get("reason", ""),
                })
                edge["raw_relation"] = edge["relation"]
                edge["relation"] = "unclear"
                edge["parent_fact_id"] = edge["child_fact_id"] = None

    # Parent->child edges do not prove that two children are disjoint.  When
    # no reviewed edge exists between siblings, add a code-owned uncertainty
    # constraint.  Analysis may then select the parent, or one supported
    # child when the parent is out of scope, but can never silently sum all
    # siblings merely because they sit below the same total.
    existing = {
        (edge["fact_a_id"], edge["fact_b_id"]): edge for edge in edges
    }
    children_by_parent = defaultdict(set)
    for edge in edges:
        if edge.get("relation") == "contains":
            children_by_parent[edge["parent_fact_id"]].add(edge["child_fact_id"])
    sibling_edges = []
    for parent_id, children in sorted(children_by_parent.items()):
        for left, right in combinations(sorted(children), 2):
            pair = (left, right)
            if pair in existing:
                continue
            edge = {
                "fact_a_id": left, "fact_b_id": right,
                "relation": "unclear", "raw_relation": "sibling_unproven",
                "parent_fact_id": None, "child_fact_id": None,
                "newer_fact_id": None, "older_fact_id": None,
                "confidence": "low", "decision_source": "sibling_conservatism",
                "evidence_raw_ids": [],
                "reason": ("both facts are children of " + parent_id
                           + "; their mutual disjointness was not established"),
                "amount_check": "not_applicable",
            }
            sibling_edges.append(edge)
            # The same two children can sit below more than one aggregate
            # parent.  Mark the synthesized pair immediately so a later
            # parent cannot emit a duplicate edge for the same Fact pair.
            existing[pair] = edge
    edges.extend(sibling_edges)

    policy = {
        "independent": "allow_joint_addition",
        "alternative_measure_of": "separate_perimeters",
        "contains": "block_joint_addition",
        "partial_overlap": "block_joint_addition",
        "revision_of": "block_joint_addition",
        "same_atomic_fact": "block_joint_addition",
        "unclear": "block_joint_addition",
    }
    for edge in edges:
        edge["joint_addition_policy"] = policy[edge["relation"]]
        edge["relation_id"] = "FR-" + sha16(
            edge["fact_a_id"], edge["fact_b_id"], edge["relation"])
    return sorted(edges, key=lambda row: (row["fact_a_id"], row["fact_b_id"]))


def run(outdir: str, judge: Judge) -> dict:
    man = Manifest(outdir, "build_fact_relations")
    names = ("facts.jsonl", "fact_state.jsonl", "entities.jsonl", "objects.jsonl",
             "fact_temporal_profile.jsonl", "observations_final.jsonl")
    for name in names:
        man.add_input(os.path.join(outdir, name))
    facts = read_jsonl(os.path.join(outdir, "facts.jsonl"))
    states = read_jsonl(os.path.join(outdir, "fact_state.jsonl"))
    profiles = read_jsonl(os.path.join(outdir, "fact_temporal_profile.jsonl"))
    observations = read_jsonl(os.path.join(outdir, "observations_final.jsonl"))
    entities = read_jsonl(os.path.join(outdir, "entities.jsonl"))
    obs_by_id = {row["raw_id"]: row for row in observations}
    entity_names = {row["id"]: str(row.get("canonical") or row["id"])
                    for row in entities}
    nodes = _build_nodes(facts, states, profiles, obs_by_id, entity_names)

    reviews = []
    # Record oversized closure blocks.  They are skipped only for the broad
    # entity key; object and exact-party keys still generate candidates.
    entity_blocks = defaultdict(list)
    oversized_facts = set()
    for node in nodes.values():
        if node.instrument in MONEY_RELATION_INSTRUMENTS and node.amount is not None:
            for entity in node.closure:
                entity_blocks[(entity, node.measure_group)].append(node.fact_id)
    for key, fact_ids in sorted(entity_blocks.items()):
        if len(fact_ids) > MAX_BLOCK_SIZE:
            oversized_facts.update(fact_ids)
            reviews.append({"reason": "oversized_entity_block",
                            "fact_ids": sorted(fact_ids),
                            "detail": {"key": list(key), "size": len(fact_ids)}})

    pairs = _candidate_pairs(nodes)
    rule_edges = []
    series = defaultdict(list)
    for node in nodes.values():
        if node.series_id:
            series[node.series_id].append(node)
    rule_pair_ids = set()
    for series_id, members in sorted(series.items()):
        ordered = sorted(members,
                         key=lambda n: (n.first_disclosed_at, n.fact_id))
        for older, newer in zip(ordered, ordered[1:]):
            pair = tuple(sorted((older.fact_id, newer.fact_id)))
            rule_pair_ids.add(pair)
            rule_edges.append({
                "fact_a_id": pair[0], "fact_b_id": pair[1],
                "relation": "revision_of", "raw_relation": "revision_of",
                "parent_fact_id": None, "child_fact_id": None,
                "newer_fact_id": newer.fact_id, "older_fact_id": older.fact_id,
                "confidence": "high", "decision_source": "guidance_series_rule",
                "evidence_raw_ids": [],
                "reason": f"successive facts in guidance series {series_id}",
            })

    by_pivot = defaultdict(list)
    for pair, score in sorted(pairs.items(), key=lambda item: (-item[1], item[0])):
        if pair in rule_pair_ids:
            continue
        if (nodes[pair[0]].series_id
                and nodes[pair[0]].series_id == nodes[pair[1]].series_id):
            # Consecutive rule edges already connect the whole revision chain.
            continue
        pivot, candidate = _orient(nodes[pair[0]], nodes[pair[1]])
        by_pivot[pivot.fact_id].append((candidate, score))

    requests = []
    validators = {}
    request_meta = {}
    for pivot_id in sorted(by_pivot):
        pivot = nodes[pivot_id]
        candidates = [item[0] for item in sorted(
            by_pivot[pivot_id], key=lambda item: (-item[1], item[0].fact_id))]
        for batch_no, batch in enumerate(_chunks(candidates, MAX_CANDIDATES_PER_PROMPT)):
            pivot_card, pivot_labels = _card(pivot, obs_by_id, "P")
            candidate_sections = []
            label_to_raw = dict(pivot_labels)
            evidence_by_candidate = {}
            for pos, candidate in enumerate(batch, 1):
                prefix = f"C{pos}"
                card, labels = _card(candidate, obs_by_id, prefix)
                label_to_raw.update(labels)
                evidence_by_candidate[candidate.fact_id] = (
                    set(pivot_labels) | set(labels))
                candidate_sections.extend([
                    f"CANDIDATE_ID={candidate.fact_id}", card,
                ])
            user = "\n\n".join([
                f"PIVOT_ID={pivot.fact_id}", "PIVOT FACT", pivot_card,
                "CANDIDATE FACTS", *candidate_sections,
            ])
            # Hash the exact rendered user input, not merely fact ids.  A card
            # formatting/election/evidence change must make an old verdict
            # unreachable even when observation membership (and fact_id) did
            # not change.
            item_id = content_id("factrel", user)
            candidate_ids = {candidate.fact_id for candidate in batch}
            requests.append((item_id, user))
            validators[item_id] = _validator(
                pivot.fact_id, candidate_ids, evidence_by_candidate)
            request_meta[item_id] = (pivot, batch, label_to_raw, batch_no)

    results = judge.batch_json(
        "r6_fact_relation", requests, RELATION_SYSTEM_PROMPT,
        validators=validators, label="fact relations") if requests else {}

    llm_edges = []
    for item_id, _user in requests:
        pivot, batch, label_to_raw, batch_no = request_meta[item_id]
        obj = results.get(item_id)
        decisions = ({d["candidate_id"]: d for d in obj["decisions"]}
                     if obj is not None else {})
        for candidate in batch:
            decision = decisions.get(candidate.fact_id)
            if decision is None:
                edge = {
                    "fact_a_id": min(pivot.fact_id, candidate.fact_id),
                    "fact_b_id": max(pivot.fact_id, candidate.fact_id),
                    "relation": "unclear", "raw_relation": "quarantine",
                    "parent_fact_id": None, "child_fact_id": None,
                    "newer_fact_id": None, "older_fact_id": None,
                    "confidence": "low", "decision_source": "llm_quarantine",
                    "evidence_raw_ids": [],
                    "reason": "missing, invalid, or quarantined relation judgment",
                }
                reviews.append({
                    "reason": "llm_relation_quarantine",
                    "fact_ids": [edge["fact_a_id"], edge["fact_b_id"]],
                    "detail": {"item_id": item_id, "batch": batch_no},
                })
            else:
                edge = _normalize_decision(
                    pivot, candidate, decision, label_to_raw)
                if edge["relation"] == "unclear":
                    reviews.append({
                        "reason": "relation_unresolved",
                        "fact_ids": [edge["fact_a_id"], edge["fact_b_id"]],
                        "detail": edge.get("reason", ""),
                    })
                if edge["raw_relation"] == "same_atomic_fact":
                    reviews.append({
                        "reason": "late_duplicate_detected",
                        "fact_ids": [edge["fact_a_id"], edge["fact_b_id"]],
                        "detail": edge.get("reason", ""),
                    })
            llm_edges.append(edge)

    edges = _finish_edges(rule_edges + llm_edges, nodes, reviews)
    by_fact = defaultdict(list)
    for edge in edges:
        by_fact[edge["fact_a_id"]].append(edge)
        by_fact[edge["fact_b_id"]].append(edge)

    uf = _UF(nodes)
    for edge in edges:
        if edge["joint_addition_policy"] != "allow_joint_addition":
            uf.union(edge["fact_a_id"], edge["fact_b_id"])
    component_members = defaultdict(list)
    for fact_id in nodes:
        component_members[uf.find(fact_id)].append(fact_id)
    component_id = {}
    for members in component_members.values():
        cid = "RC-" + sha16(*sorted(members))
        for fact_id in members:
            component_id[fact_id] = cid

    statuses = []
    for fact_id in sorted(nodes):
        related = by_fact.get(fact_id, [])
        unresolved = sum(edge["relation"] == "unclear" for edge in related)
        quarantined = sum(edge["decision_source"] == "llm_quarantine"
                          for edge in related)
        statuses.append({
            "fact_id": fact_id,
            "candidate_count": len(related),
            "resolved_count": len(related) - unresolved,
            "unresolved_count": unresolved,
            "quarantine_count": quarantined,
            "coverage_complete": quarantined == 0 and fact_id not in oversized_facts,
            "relation_component_id": component_id[fact_id],
        })

    paths = {
        "relations": os.path.join(outdir, "fact_relations.jsonl"),
        "status": os.path.join(outdir, "fact_relation_status.jsonl"),
        "reviews": os.path.join(outdir, "fact_relation_reviews.jsonl"),
    }
    write_jsonl(edges, paths["relations"])
    write_jsonl(statuses, paths["status"])
    write_jsonl(reviews, paths["reviews"])
    for path in paths.values():
        man.add_output(path)

    stats = {
        "facts": len(nodes),
        "candidate_pairs": len(pairs),
        "rule_relations": len(rule_edges),
        "llm_pairs": len(llm_edges),
        "llm_prompt_batches": len(requests),
        "contains": sum(edge["relation"] == "contains" for edge in edges),
        "partial_overlap": sum(edge["relation"] == "partial_overlap" for edge in edges),
        "revision": sum(edge["relation"] == "revision_of" for edge in edges),
        "alternative_measure": sum(edge["relation"] == "alternative_measure_of"
                                   for edge in edges),
        "independent": sum(edge["relation"] == "independent" for edge in edges),
        "same_atomic_fact": sum(edge["relation"] == "same_atomic_fact" for edge in edges),
        "unresolved": sum(edge["relation"] == "unclear" for edge in edges),
        "reviews": len(reviews),
    }
    man.done(stats, judge.report())
    return stats
