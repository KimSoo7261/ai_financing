# AICapitalMap `pipeline_new` — Fact Registry v2

`pipeline_new` is the actively developed aggregation-ready pipeline. The
unchanged predecessor is `../pipeline_clean`; never point a v2 run at a frozen
v1 output directory.

The current release boundary is the **Fact layer**. A normal command runs
every Fact-level stage **through `validate`** (coverage, content-id, ledger
replay, and temporal-profile checks on the Fact table itself), while the
deferred stages — `build_fact_relations` and `build_fact_views` — run only
when explicitly requested (`--relations` / `--views`, or named via
`--from/--to-stage`); their validation checks are marked skipped when their
artifacts are absent. The new observations, typed objects, and Facts are
reviewed before the deferred Fact-relation/aggregation design is activated.

## Default data flow

> **No corpus ships with this repository.** The chunk sets are licensed filings
> text and are excluded by `.gitignore`. Point `--chunks` at your own `.jsonl`,
> or set `AICAP_DATA_DIR` to the directory holding it. A missing corpus fails
> immediately with the path that was resolved, rather than deep inside the first
> stage:
>
> ```
> chunk set not found: .../chunks_full_v8.1.jsonl
>   the chunk corpora are licensed filings text and are not
>   included in this repository. Point --chunks at your own
>   .jsonl, or set AICAP_DATA_DIR to the directory holding it.
> ```

```text
chunks.jsonl
  screen                     relevance screen (or seed pre-screened input)
  extract                    strict v2 atomic-observation extraction
  audit                      compact eight-check independent audit
      -> capital_instances.jsonl + audit.jsonl
  intake                     schema-version gate + immutable observation store
  normalize_amounts          deterministic-first amount parsing
  resolve_entities_objects   entity registry + typed object registry
  extract_time               fact-local T1–T4 time claims
  quality_gate               joined observation status
  deduplicate_facts          online observation -> Fact, tiered Top-5 retrieval
  build_fact_state           elected/union Fact fields + provenance
  build_temporal_profile     authoritative Fact timeline
  validate                   Fact-level gate (relation/view checks skip)  <-- default stop
```

`build_fact_relations` (--relations) and `build_fact_views` (--views) remain
explicitly requested diagnostics, but Fact Relation v2 is not accepted as
finalized research logic yet.  The CLI and the Python API share one boundary
definition (enforced in the orchestrator), and the default run's exit code
reflects the validation verdict.

## Run

```powershell
# Production (uses the configured model profile; stops at Fact level)
py -3 run_pipeline.py --set full --outdir run_v2_full

# Small no-cost wiring run
py -3 run_pipeline.py --mock --set dirtest10 --no-screen --limit 10 `
  --outdir smoke_v2 --no-batch

# Explicitly rebuild a stage range
py -3 run_pipeline.py --outdir run_v2_full `
  --from-stage deduplicate_facts --to-stage build_temporal_profile

# Compatibility-only later-stage checks after reviewing the Fact output
py -3 run_pipeline.py --mock --outdir smoke_v2 `
  --from-stage build_fact_relations --to-stage validate --no-batch
```

`--source-dir` downstream-only intake accepts only rows stamped
`schema_version="fact_registry_v2"`. An unversioned/v1 Stage-3 snapshot fails
loudly because the new object, amount-level, provenance, and debt fields cannot
be recovered honestly downstream. Rerun `extract` + `audit` under this tree.

## v2 contracts that matter

- Direction is still decided by the extraction LLM and independently audited
  in the existing `role_resolution/checks/note` structure. Deterministic code
  compares/routes and applies only two-model-consensus repairs (the exact
  two-signal swap, the consensus counterparty clear, the degenerate-slot fix)
  or a signed human override. Human overrides are bound to the reviewed row's
  content digest (a re-extracted row re-queues instead of replaying a stale
  decision) and are re-checked for degenerate slots after application. Dedup
  never repairs direction.
- Cache identity is content-derived end to end: screen/extract keys fold in a
  digest of the chunk TEXT, the Stage-3 review key digests the auditor-visible
  row, and the dedup pair key digests the rendered cards. Reads are probed
  per current chunk/row, so a reused outdir can never union a stale corpus,
  a superseded prompt's verdicts, or another model's output into this run —
  and `--limit` counts only chunks that exist in the current corpus.
- `reuters_news` is news everywhere (`config.is_news_source`): the prompt
  header, the null-target default (a null-target Reuters row stays null and
  gates out), and `--no-news` all share one predicate.
- The gate blocks (review_required) every amount-fallback failure
  (`llm_all_null`, `*_det_fallback`, `rule_not_reconstructible`) and rows
  whose Stage-3 audit never arrived; `amount_value_source` says who actually
  produced the kept numbers. Fact-state elections are status-aware: a clean
  member's amount beats a flagged higher-rank member's, capacity is elected
  (not anchor-copied), and `ai_infra_scope` voters are audit-disqualified
  like every other audited field.
- Each observation has at most one intact `object_name`, plus type/evidence and
  optional parent context. Different `object_type` values never merge. Parent
  objects are lookup-only and materialize only for explicit, type-compatible
  `component_of` evidence.
- `amount_basis`, `amount_bound_type`, and `amount_aggregation_level` are
  orthogonal. Valuation, transaction consideration, financing capacity,
  period flow, tranche/component, and rate semantics are no longer collapsed
  onto one axis.
- Fact candidate retrieval is deterministic Top 5. Only composite keys can
  nominate a candidate; fixed tiers are ranked lexicographically, then exact
  pair rules/LLM judgment decide sameness.
- Structured output distinguishes `ok`, `partial`, and `quarantine`.
  Provider-confirmed token cutoffs trigger deterministic split-and-rerun;
  parse/schema failures get a lenient (`strict=False`) re-parse first — the
  fix that recovers every structural failure observed on the real 2025 runs —
  then one repair attempt, and are never mislabeled as successful salvage.
  A quarantined extraction whose payload carries an amount ≥
  `AICAP_RESCUE_MIN_USD` (default $10B) gets ONE additional fix-JSON-only
  repair round on the failed raw text, cached under its own namespace with
  any outcome so the rescue is structurally one-shot across runs. Quarantined
  chunk ids are listed in the run report.
- Every stage manifest records input/output hashes, and each invocation
  checks non-selected stages' manifests against the files now on disk —
  a mixed-state directory prints a loud `stale_stages` warning and records it
  in `report.json`; `reports_log.jsonl` keeps an append-only record of every
  invocation into the directory.

## Documentation and verification

- [`IMPLEMENTATION_MANUAL_V2.md`](IMPLEMENTATION_MANUAL_V2.md): exhaustive
  before/after implementation record.
- [`FIELD_STAGE_MATRIX_V2.md`](FIELD_STAGE_MATRIX_V2.md): producer/consumer
  ownership for every new or changed field.
- [`LEGACY_V1_TO_V2_CROSSWALK.md`](LEGACY_V1_TO_V2_CROSSWALK.md): explicit
  legacy interpretation limits.
- [`schema_manifest_v2.json`](schema_manifest_v2.json): machine-readable enums,
  stage boundary, artifacts, and compatibility policy.
- `tests/test_v2_contract.py`: strict schema, direction, object, Top-5,
  recovery, and migration-gate regression tests.

Run the local gates with:

```powershell
py -3 -m compileall -q aicap tools tests
py -3 -m unittest discover -s tests -v
```

Prompt edits automatically change the prompt-hash cache namespace. Registry
stage names and v2 schema version likewise keep incompatible results from
being replayed as current output.
