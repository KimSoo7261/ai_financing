#!/usr/bin/env python3
"""Stage `validate` -- coverage, link-ledger replay, and view purity checks.

What each check DOES and DOES NOT prove:
  cover_facts       every non-quarantined observation sits in exactly one Fact
  content_ids       every fact_id equals the hash of its sorted members
  replay_facts      re-running the union-find over fact_links.jsonl reproduces
                    the shipped partition.  This is a SELF-CONSISTENCY replay
                    of the ledger this run wrote -- it catches bookkeeping
                    bugs, NOT wrong merge judgments (a bad LLM verdict
                    replays "consistently" too)
  fact_views        the view builder's own invariants were green
  asof_purity       independently re-checks that known views contain no
                    evidence disclosed after quarter end
  fail_closed       independently re-checks that no multi-Fact measure cell
                    carries a value
  economic_support  independently re-checks economic rows' interval overlap
  fact_relations    relation endpoints/schema/policies are internally valid
  relation_coverage every Fact has one status row whose degree matches edges
  containment_dag  contains edges form a directed acyclic graph

validation.success=true therefore means "internally consistent and no
look-ahead", not "the Fact partition is correct".  Partition quality is only
measurable against an independent reference run or human adjudication.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque

from .common import Manifest, q_bounds, read_jsonl, sha16, write_json


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra

    def groups(self):
        out = defaultdict(set)
        for x in self.p:
            out[self.find(x)].add(x)
        return {frozenset(v) for v in out.values()}


def _read_optional(path: str):
    """(rows, present).  The relation artifacts are produced by a DELIBERATELY
    deferred stage (DEC-027): a Fact-level run validates without them and the
    relation checks are marked skipped, exactly like the held-out views."""
    if not os.path.exists(path):
        return [], False
    return read_jsonl(path), True


def run(outdir: str) -> dict:
    man = Manifest(outdir, "validate")
    input_names = (
        "observations_final.jsonl",
        "facts.jsonl",
        "fact_members.jsonl",
        "fact_links.jsonl",
        "fact_temporal_profile.jsonl",
        "fact_relations.jsonl",
        "fact_relation_status.jsonl",
    )
    for name in input_names:
        man.add_input(os.path.join(outdir, name))

    obs = read_jsonl(os.path.join(outdir, "observations_final.jsonl"))
    facts = read_jsonl(os.path.join(outdir, "facts.jsonl"))
    members = read_jsonl(os.path.join(outdir, "fact_members.jsonl"))
    flinks = read_jsonl(os.path.join(outdir, "fact_links.jsonl"))
    profiles = read_jsonl(os.path.join(outdir, "fact_temporal_profile.jsonl"))
    relations, relations_built = _read_optional(
        os.path.join(outdir, "fact_relations.jsonl"))
    relation_status, _rs_built = _read_optional(
        os.path.join(outdir, "fact_relation_status.jsonl"))
    relations_built = relations_built and _rs_built
    checks: dict[str, dict] = {}

    # Every admitted observation belongs to exactly one Fact.
    eligible = {
        row["raw_id"]
        for row in obs
        if row["observation_status"] != "quarantined"
    }
    in_facts = [row["raw_id"] for row in members]
    duplicated = (
        {rid for rid in in_facts if in_facts.count(rid) > 1}
        if len(in_facts) != len(set(in_facts))
        else set()
    )
    missing = eligible - set(in_facts)
    unexpected = set(in_facts) - eligible
    checks["cover_facts"] = {
        "ok": not duplicated and not missing and not unexpected,
        "missing": sorted(missing),
        "duplicated": sorted(duplicated),
        "unexpected": sorted(unexpected),
    }

    # Time independence (2026-08-19 ruling): no signal originating in the
    # time layer may change observation_status.  Time imperfection degrades
    # a fact's temporal_quality (floor: disclosure_only), never its standing.
    time_blocked = []
    for row in obs:
        for reason in row.get("blocking_reasons") or []:
            if reason == "time_needs_review" or (
                    reason.startswith("audit_bad:")
                    and "time_ok" in reason.split(":", 1)[1].split(",")):
                time_blocked.append((row["raw_id"], reason))
    checks["time_independence"] = {
        "ok": not time_blocked,
        "violations": time_blocked[:20],
    }

    # Fact identity is content-addressed over observation membership.
    bad_ids = [
        fact["fact_id"]
        for fact in facts
        if fact["fact_id"] != "RF-" + sha16(*sorted(fact["raw_ids"]))
    ]
    checks["content_ids"] = {
        "ok": not bad_ids,
        "bad_fact_ids": bad_ids,
    }

    # Replaying the Fact link ledger must reproduce the shipped partition.
    uf = _UF()
    for rid in eligible:
        uf.find(rid)
    for link in flinks:
        if link["decision"] in ("same", "same_bridge_merge"):
            for member in link.get("fact_members") or []:
                uf.union(link["raw_id"], member)
    replay = uf.groups()
    shipped = {frozenset(fact["raw_ids"]) for fact in facts}
    checks["replay_facts"] = {
        "ok": replay == shipped,
        "replay_n": len(replay),
        "shipped_n": len(shipped),
    }

    # The default run's terminal product gets its own coverage check: exactly
    # one temporal-profile row per shipped Fact, no orphan profiles.
    prof_ids = [row.get("fact_id") for row in profiles]
    prof_set = set(prof_ids)
    fact_id_set = {fact["fact_id"] for fact in facts}
    checks["profile_coverage"] = {
        "ok": (len(prof_ids) == len(prof_set)
               and prof_set == fact_id_set),
        "missing_profiles": sorted(fact_id_set - prof_set)[:20],
        "orphan_profiles": sorted(prof_set - fact_id_set)[:20],
        "duplicate_profiles": len(prof_ids) - len(prof_set),
    }

    # Fact relations are a typed sparse graph, never a Deal registry.  Every
    # endpoint must be a shipped Fact, every candidate pair appears once, and
    # the code-owned joint-addition policy must agree with the relation type.
    fact_ids = {fact["fact_id"] for fact in facts}
    relation_values = {
        "contains", "partial_overlap", "revision_of",
        "alternative_measure_of", "independent", "same_atomic_fact", "unclear",
    }
    expected_policy = {
        "independent": "allow_joint_addition",
        "alternative_measure_of": "separate_perimeters",
        "contains": "block_joint_addition",
        "partial_overlap": "block_joint_addition",
        "revision_of": "block_joint_addition",
        "same_atomic_fact": "block_joint_addition",
        "unclear": "block_joint_addition",
    }
    seen_pairs = set()
    relation_bad = []
    degree = defaultdict(int)
    contains_graph = defaultdict(set)
    contains_indegree = defaultdict(int)
    contains_nodes = set()
    for row in relations:
        a, b = row.get("fact_a_id"), row.get("fact_b_id")
        pair = tuple(sorted((str(a), str(b)))) if a and b else None
        relation = row.get("relation")
        reasons = []
        if not pair or a == b or a not in fact_ids or b not in fact_ids:
            reasons.append("bad_endpoint")
        if pair in seen_pairs:
            reasons.append("duplicate_pair")
        if relation not in relation_values:
            reasons.append("bad_relation")
        elif row.get("joint_addition_policy") != expected_policy[relation]:
            reasons.append("bad_policy")
        if relation == "contains":
            parent, child = row.get("parent_fact_id"), row.get("child_fact_id")
            if {parent, child} != {a, b} or parent == child:
                reasons.append("bad_containment_direction")
            else:
                if child not in contains_graph[parent]:
                    contains_graph[parent].add(child)
                    contains_indegree[child] += 1
                    contains_indegree.setdefault(parent, 0)
                    contains_nodes.update((parent, child))
        if reasons:
            relation_bad.append((a, b, reasons))
        if pair:
            seen_pairs.add(pair)
            degree[a] += 1
            degree[b] += 1
    checks["fact_relations"] = {
        "ok": not relation_bad,
        "violations": relation_bad[:50],
    }

    status_by = defaultdict(list)
    for row in relation_status:
        status_by[row.get("fact_id")].append(row)
    coverage_bad = []
    if None in status_by:
        coverage_bad.append((None, "missing_fact_id", len(status_by[None])))
    status_fact_ids = {fact_id for fact_id in status_by if fact_id is not None}
    for fact_id in sorted(fact_ids | status_fact_ids):
        rows = status_by.get(fact_id, [])
        if len(rows) != 1:
            coverage_bad.append((fact_id, "status_row_count", len(rows)))
            continue
        row = rows[0]
        if int(row.get("candidate_count", -1)) != degree.get(fact_id, 0):
            coverage_bad.append((fact_id, "candidate_degree_mismatch",
                                 row.get("candidate_count"), degree.get(fact_id, 0)))
        if not isinstance(row.get("coverage_complete"), bool):
            coverage_bad.append((fact_id, "coverage_not_boolean",
                                 row.get("coverage_complete")))
    checks["relation_coverage"] = {
        "ok": not coverage_bad,
        "violations": coverage_bad[:50],
    }

    queue = deque(sorted(node for node in contains_nodes
                         if contains_indegree[node] == 0))
    visited = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for child in sorted(contains_graph[node]):
            contains_indegree[child] -= 1
            if contains_indegree[child] == 0:
                queue.append(child)
    cycle_nodes = sorted(contains_nodes - visited)
    checks["containment_dag"] = {
        "ok": not cycle_nodes,
        "cycle_nodes": cycle_nodes[:50],
    }

    # Relation artifacts absent -> the three relation checks are SKIPPED, not
    # failed (mirrors the held-out views below): the relation stage is a
    # deferred, explicitly requested diagnostic (DEC-027), and the Fact-level
    # release boundary must still get a validation verdict without it.
    if not relations_built:
        for name in ("fact_relations", "relation_coverage", "containment_dag"):
            checks[name] = {
                "ok": True, "skipped": True,
                "note": ("fact_relations not built (deferred per DEC-027; "
                         "run build_fact_relations explicitly)"),
            }

    # The view builder's own invariants must be green.
    view_root = os.path.join(outdir, "fact_views")
    view_report_path = os.path.join(view_root, "fact_views_report.json")
    man.add_input(view_report_path)
    if os.path.exists(view_report_path):
        with open(view_report_path, encoding="utf-8") as fh:
            view_report = json.load(fh)
        failed = sorted(
            name
            for name, result in (view_report.get("invariants") or {}).items()
            if not result.get("ok")
        )
        checks["fact_views"] = {
            "ok": bool(view_report.get("success")) and not failed,
            "failed": failed,
        }
    else:
        # 2026-08-19 scope decision: the fact table is the product; quarterly
        # views are HELD OUT of the default run (rebuild with --views).  An
        # absent report is therefore a skipped stage, not a failure -- the
        # three view-replay checks below go vacuous for the same reason
        # (quarters=[] without a report) and are marked skipped alongside.
        checks["fact_views"] = {
            "ok": True,
            "skipped": True,
            "note": "fact_views not built (held out; run with --views)",
        }

    # Independently replay the three load-bearing quarterly guarantees.
    obs_by_id = {row["raw_id"]: row for row in obs}
    fail_closed_bad = []
    asof_bad = []
    economic_bad = []
    quarters = []
    if os.path.isdir(view_root):
        quarters = sorted(
            filename[len("known_facts_"):-len(".jsonl")]
            for filename in os.listdir(view_root)
            if filename.startswith("known_facts_")
            and filename.endswith(".jsonl")
        )
    for quarter in quarters:
        q_start, q_end = q_bounds(quarter)
        known_path = os.path.join(
            view_root, f"known_facts_{quarter}.jsonl"
        )
        for row in read_jsonl(known_path):
            raw_ids = row.get("known_raw_ids") or []
            late = [
                rid
                for rid in raw_ids
                if str(obs_by_id.get(rid, {}).get("doc_date") or "9999")
                > q_end
            ]
            source_id = row.get("asof_source_raw_id")
            source_late = (
                source_id
                and str(
                    obs_by_id.get(source_id, {}).get("doc_date") or "9999"
                )
                > q_end
            )
            allowed_targets = {
                target
                for rid in raw_ids
                for target in (
                    obs_by_id.get(rid, {}).get("target_entity_ids") or []
                )
            }
            leaked_targets = (
                set(row.get("target_entity_ids_asof") or [])
                - allowed_targets
            )
            if late or source_late or leaked_targets:
                asof_bad.append(
                    (
                        quarter,
                        row.get("fact_id"),
                        sorted(late),
                        bool(source_late),
                        sorted(leaked_targets),
                    )
                )

        measure_path = os.path.join(
            view_root, f"target_quarter_measures_{quarter}.jsonl"
        )
        for row in read_jsonl(measure_path):
            if (
                len(row.get("candidate_fact_ids") or []) > 1
                and row.get("value") is not None
            ):
                fail_closed_bad.append(
                    (quarter, row.get("target_key"), row.get("measure_family"))
                )

        economic_path = os.path.join(
            view_root, f"economic_facts_{quarter}.jsonl"
        )
        for row in read_jsonl(economic_path):
            valid_from = row.get("valid_from")
            valid_to = row.get("valid_to")
            if (
                not valid_from
                or valid_from > q_end
                or (valid_to and valid_to < q_start)
            ):
                economic_bad.append(
                    (quarter, row.get("fact_id"), valid_from, valid_to)
                )

    views_built = os.path.exists(view_report_path)
    checks["fail_closed"] = {
        "ok": not fail_closed_bad,
        "violations": fail_closed_bad,
        **({} if views_built else {"skipped": True}),
    }
    checks["asof_purity"] = {
        "ok": not asof_bad,
        "violations": asof_bad,
        **({} if views_built else {"skipped": True}),
    }
    checks["economic_support"] = {
        "ok": not economic_bad,
        "violations": economic_bad,
        **({} if views_built else {"skipped": True}),
    }

    success = all(check["ok"] for check in checks.values())
    result = {"success": success, "checks": checks}
    output_path = os.path.join(outdir, "validation.json")
    write_json(result, output_path)
    man.add_output(output_path)
    man.done({"success": success, **{
        name: check["ok"] for name, check in checks.items()
    }})
    return result
