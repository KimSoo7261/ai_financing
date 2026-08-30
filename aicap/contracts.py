"""Canonical stage names and artifact ownership.

This module is deliberately free of pipeline implementation imports.  It is
the small, stable contract that the CLI, orchestrator, manifests, tests, and
documentation share.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageContract:
    name: str
    description: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


STAGE_CONTRACTS = (
    StageContract(
        "screen",
        "deterministic pre-screen plus relevance screening",
    ),
    StageContract(
        "extract",
        "extract atomic capital observations from relevant chunks",
        produces=("raw_stage2.jsonl",),
    ),
    StageContract(
        "audit",
        "role-aware audit and Stage-3 admission",
        requires=("raw_stage2.jsonl",),
        produces=(
            "capital_instances.jsonl",
            "audit.jsonl",
            "dropped.jsonl",
            "direction_review_queue.jsonl",
        ),
    ),
    StageContract(
        "intake",
        "copy immutable Stage-3 observations into the registry run",
        # intake's real inputs (capital_instances.jsonl + audit.jsonl) live in
        # the SOURCE directory, not the run directory, so the outdir-rooted
        # contract checker cannot declare them; intake itself fails loudly
        # when they are missing.
        produces=("observations.jsonl", "intake_quarantine.jsonl"),
    ),
    StageContract(
        "normalize_amounts",
        "deterministic-first amount normalization",
        requires=("observations.jsonl",),
        produces=("obs_amounts.jsonl",),
    ),
    StageContract(
        "resolve_entities_objects",
        "sequential entity and typed-object registry resolution (v2)",
        requires=("observations.jsonl",),
        produces=(
            "entities.jsonl",
            "objects.jsonl",
            "entity_links.jsonl",
            "object_links.jsonl",
            "observations_resolved.jsonl",
        ),
    ),
    StageContract(
        "extract_time",
        "observation-level time extraction (gate/read/check/resolve)",
        requires=("observations.jsonl", "observations_resolved.jsonl"),
        produces=("obs_time_claims.jsonl", "obs_time_flags.jsonl"),
    ),
    StageContract(
        "quality_gate",
        "join normalized fields and assign observation quality status",
        requires=(
            "observations.jsonl",
            "obs_amounts.jsonl",
            "observations_resolved.jsonl",
            "obs_time_flags.jsonl",
        ),
        produces=("observations_final.jsonl",),
    ),
    StageContract(
        "deduplicate_facts",
        "bounded candidate retrieval and observation-to-fact linking",
        requires=("observations_final.jsonl", "obs_time_claims.jsonl"),
        produces=(
            "facts.jsonl",
            "fact_members.jsonl",
            "fact_links.jsonl",
            "fact_reviews.jsonl",
            "guidance_series.jsonl",
        ),
    ),
    StageContract(
        "build_fact_state",
        "deterministic Fact participants and amount evidence",
        requires=(
            "facts.jsonl",
            "observations_final.jsonl",
        ),
        produces=(
            "fact_state.jsonl",
            "fact_participants.jsonl",
        ),
    ),
    StageContract(
        "build_temporal_profile",
        "authoritative Fact temporal profile",
        requires=(
            "facts.jsonl",
            "observations_final.jsonl",
            "obs_time_claims.jsonl",
            "obs_time_flags.jsonl",
        ),
        produces=("fact_temporal_profile.jsonl", "fact_profile_flags.jsonl"),
    ),
    StageContract(
        "build_fact_relations",
        "typed Fact-to-Fact overlap and containment graph",
        # objects.jsonl is deliberately NOT declared: the deferred relation
        # stage (DEC-027) does not read the object hierarchy yet -- declaring
        # it implied a hierarchy participation that does not exist.
        requires=(
            "facts.jsonl",
            "fact_state.jsonl",
            "entities.jsonl",
            "fact_temporal_profile.jsonl",
            "observations_final.jsonl",
        ),
        produces=(
            "fact_relations.jsonl",
            "fact_relation_status.jsonl",
            "fact_relation_reviews.jsonl",
        ),
    ),
    StageContract(
        "build_fact_views",
        "quarterly Fact views",
        requires=(
            "facts.jsonl",
            "fact_temporal_profile.jsonl",
            "fact_participants.jsonl",
            "observations_final.jsonl",
            "obs_time_claims.jsonl",
            "obs_time_flags.jsonl",
        ),
        # The per-quarter tables (known_facts_<q>.jsonl, ...) carry dynamic
        # names and cannot be declared statically; the report's `quarters`
        # block records what was actually written.
        produces=("fact_views/fact_views_report.json",),
    ),
    StageContract(
        "validate",
        "cross-stage replay and view-contract validation",
        # Only the FACT-LEVEL artifacts are required, so validate runs at the
        # DEC-027 release boundary ("Fact construction and its
        # validation/reporting gates").  The relation files and
        # fact_views_report.json are deliberately NOT required: relations are
        # deferred and the quarterly views are held out of the default run
        # (2026-08-19 scope decision); validation marks the relation and view
        # checks skipped when their artifacts are absent.
        requires=(
            "observations_final.jsonl",
            "facts.jsonl",
            "fact_members.jsonl",
            "fact_links.jsonl",
            "fact_temporal_profile.jsonl",
        ),
        produces=("validation.json",),
    ),
)

CONTRACT_BY_NAME = {stage.name: stage for stage in STAGE_CONTRACTS}
CANONICAL_STAGE_NAMES = tuple(stage.name for stage in STAGE_CONTRACTS)

REGISTRY_STAGE_NAMES = (
    "intake",
    "normalize_amounts",
    "resolve_entities_objects",
    "extract_time",
    "quality_gate",
    "deduplicate_facts",
    "build_fact_state",
    "build_temporal_profile",
    "build_fact_relations",
    "build_fact_views",
    "validate",
)


def normalize_stage_name(name: str) -> str:
    normalized = str(name).strip()
    if normalized not in CONTRACT_BY_NAME:
        raise ValueError(
            f"unknown stage {name!r}; expected one of {sorted(CONTRACT_BY_NAME)}"
        )
    return normalized


def select_stages(
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[str, ...]:
    order = CANONICAL_STAGE_NAMES
    start = normalize_stage_name(from_stage or order[0])
    end = normalize_stage_name(to_stage or order[-1])
    i, j = order.index(start), order.index(end)
    if i > j:
        raise ValueError(f"from-stage {start!r} occurs after to-stage {end!r}")
    return order[i : j + 1]


def artifact_owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    for stage in STAGE_CONTRACTS:
        for artifact in stage.produces:
            previous = owners.get(artifact)
            if previous:
                raise AssertionError(
                    f"artifact {artifact!r} has two owners: {previous}, {stage.name}"
                )
            owners[artifact] = stage.name
    return owners


ARTIFACT_OWNERS = artifact_owners()


class ContractViolation(RuntimeError):
    """Raised when a stage's declared file contract is not satisfied."""


def assert_artifacts_exist(
    root: str,
    artifacts: Iterable[str],
    *,
    stage: str,
    relation: str,
) -> None:
    missing = [
        artifact
        for artifact in artifacts
        if not os.path.exists(os.path.join(root, artifact))
    ]
    if missing:
        raise ContractViolation(
            f"{stage}: missing declared {relation}: {', '.join(missing)}"
        )


def assert_stage_inputs(stage: str, root: str) -> None:
    contract = CONTRACT_BY_NAME.get(stage)
    if contract is not None:
        assert_artifacts_exist(
            root, contract.requires, stage=stage, relation="inputs"
        )


def assert_stage_outputs(stage: str, root: str) -> None:
    contract = CONTRACT_BY_NAME.get(stage)
    if contract is not None:
        assert_artifacts_exist(
            root, contract.produces, stage=stage, relation="outputs"
        )
