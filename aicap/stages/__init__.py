"""The Fact-registry stages.

Each module implements one canonical stage as ``run(...) -> stats`` and writes
its declared artifacts plus a manifest into the run directory:

    intake            Stage-3 observations -> immutable observation store
    amounts           deterministic-first amount normalization
    entity_registry   two-pass entity / project resolution
    time_extraction   observation-level time claims (delegates to
                      aicap.time_normalizer)
    quality_gate      join normalized fields, assign observation status
    fact_dedup        observation -> Fact linking (tiered resolution)
    fact_state        per-Fact participants and amount evidence
    temporal_profile  authoritative per-Fact temporal profile
    fact_relations    typed Fact containment / overlap / revision graph
    fact_views        quarterly known / disclosure / economic / measure /
                      network views
    validation        cross-stage replay and view-contract checks

Shared plumbing lives in ``common`` (IO, buckets, manifest) and ``judge``
(the cached LLM adjudicator).
"""
