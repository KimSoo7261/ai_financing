"""Stage sequencing for the whole pipeline.

One entry (`run`) sequences extraction (screen -> extract -> audit) and the
Fact registry (intake -> ... -> validate) over a single run directory, with
per-stage input/output contract checks and one consolidated report.json.

A complete run reads Stage-3 artifacts from its own run directory.  A
downstream-only run may read an immutable Stage-3 snapshot from
options.source_dir while writing every registry artifact to a new directory.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import extraction
from . import paths as PC
from .config import SCHEMA_VERSION
from .contracts import (REGISTRY_STAGE_NAMES, assert_stage_inputs,
                        assert_stage_outputs, select_stages)
from .stages import (amounts, entity_registry, fact_dedup, fact_relations,
                     fact_state, fact_views, intake, quality_gate,
                     temporal_profile, time_extraction, validation)
from .progress import fmt_secs, print_run_table
from .stages.common import REG_MODEL, file_sha, write_json
from .stages.judge import Judge


@dataclass(frozen=True)
class RunOptions:
    outdir: str
    chunks_path: str | None = None
    source_dir: str | None = None
    set_id: str | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    mock: bool = False
    batch: bool = True
    screen: bool | None = None
    limit: int | None = None
    include_news: bool = True
    # 2026-08-19 scope decision: the FACT TABLE is the product; the quarterly
    # view builder is held out of the default run.  Opt back in with --views
    # (or by naming build_fact_views explicitly via --from/--to-stage).
    views: bool = False
    # DEC-027: the Fact-relation stage is deferred; it runs only when
    # explicitly requested (--relations, or named via --from/--to-stage).
    relations: bool = False


def _needs_chunks(selected: tuple[str, ...]) -> bool:
    return any(name in selected for name in ("screen", "extract", "audit"))


def _run_extraction_stage(name: str, options: RunOptions, chunks_path: str) -> dict:
    if name == "screen":
        prescreened = PC.is_prescreened(options.set_id) if options.set_id else False
        do_screen = options.screen if options.screen is not None else not prescreened
        if do_screen:
            return extraction.run_stage1(
                chunks_path,
                options.outdir,
                mock=options.mock,
                limit=options.limit,
                batch=options.batch,
            )
        extraction.seed_all_relevant(
            chunks_path,
            options.outdir,
            limit=options.limit,
        )
        return {"mode": "seed_all_relevant"}
    if name == "extract":
        return extraction.run_stage2(
            chunks_path,
            options.outdir,
            mock=options.mock,
            batch=options.batch,
            limit=options.limit,
        )
    if name == "audit":
        return extraction.run_stage3(
            chunks_path,
            options.outdir,
            mock=options.mock,
            include_news=options.include_news,
            batch=options.batch,
        )
    raise ValueError(f"not an extraction stage: {name}")


def make_registry_mock_fn():
    """Canned Judge answers for --mock registry runs, dispatched on a stable
    phrase in each system prompt.  The canned FACT REGISTRY answer is always
    'different_fact', so a mock run exercises zero LLM-tier merges: the
    deterministic rule tiers still merge, but the Fact partition over-splits
    relative to a real run.  Mock output is wiring-test output only."""
    def mock_fn(system, user, cfg):
        if "FACT RELATION JUDGE" in system:
            pivot = next((line.split("=", 1)[1].strip()
                          for line in user.splitlines()
                          if line.startswith("PIVOT_ID=")), "mock-pivot")
            candidates = [line.split("=", 1)[1].strip()
                          for line in user.splitlines()
                          if line.startswith("CANDIDATE_ID=")]
            return json.dumps({
                "pivot_id": pivot,
                "decisions": [
                    {"candidate_id": candidate, "relation": "unclear",
                     "confidence": "low", "evidence_ids": [],
                     "reason": "mock"}
                    for candidate in candidates
                ],
            })
        if "AMOUNT NORMALIZER" in system:
            return json.dumps({"amount_point": None, "amount_low": None,
                               "amount_high": None, "bound_type": "unclear",
                               "period_scope": "unclear"})
        if "REGISTRY JUDGE" in system and "choice" in system:
            return json.dumps({"choice": -1})
        if "FACT REGISTRY JUDGE" in system:
            return json.dumps({"decision": "different_fact",
                               "reason": "mock", "confidence": "low"})
        return "{}"
    return mock_fn


def run_registry(
    *,
    indir: str,
    outdir: str,
    stages,
    mock: bool = False,
    batch: bool = True,
) -> dict:
    """Run the selected registry stages in canonical order over one run
    directory.  `indir` is where intake reads Stage-3 artifacts (== outdir
    for a complete run) and where extract_time warms its cache from.

    `batch` reaches the three batching registry stages (amounts, entity /
    typed-object registry, time); the fact-pair judge is sequential by design
    (union-find ordering) and is unaffected either way."""
    wanted = tuple(stages)
    judge = Judge(outdir, mock=mock,
                  mock_fn=make_registry_mock_fn() if mock else None,
                  batch=batch)

    print(f"registry: {indir} -> {outdir}  model={REG_MODEL} mock={mock} batch={batch}")
    result: dict = {"stages": {}}

    def _judge_snap():
        rep = judge.report()
        return (int(rep.get("_total_calls", 0) or 0),
                int(rep.get("_total_cache_hits", 0) or 0),
                float(rep.get("_est_cost", 0) or 0))

    def go(name, fn, *fargs):
        if name not in wanted:
            return
        print(f"-- {name} ...", flush=True)
        t0 = time.monotonic()
        c0, h0, e0 = _judge_snap()
        assert_stage_inputs(name, outdir)
        stats = fn(*fargs)
        assert_stage_outputs(name, outdir)
        elapsed = time.monotonic() - t0
        c1, h1, e1 = _judge_snap()
        brief = {k: v for k, v in stats.items()
                 if isinstance(v, (int, float, str, bool))}
        print(f"   {name}: {brief}")
        print(f"   {name}: done in {fmt_secs(elapsed)} | judge +{c1 - c0} calls "
              f"(+{h1 - h0} cached) ~+{e1 - e0:.2f}", flush=True)
        stats["elapsed_s"] = round(elapsed, 1)
        stats["judge_delta"] = {"calls": c1 - c0, "cache_hits": h1 - h0,
                                "est_cost": round(e1 - e0, 4)}
        result["stages"][name] = stats

    # Stage-3 output -> immutable observation store
    go("intake", intake.run, indir, outdir)
    # deterministic-first amount normalization
    go("normalize_amounts", amounts.run, outdir, judge)
    # sequential entity + typed-object registry and observation backfill
    go("resolve_entities_objects", entity_registry.run, outdir, judge)
    # T1-T4 observation-level time claims (warms its cache from indir)
    go("extract_time", time_extraction.run, outdir, indir, mock, batch)
    # join normalized fields, stamp observation status
    go("quality_gate", quality_gate.run, outdir)
    # observation -> Fact linking
    go("deduplicate_facts", fact_dedup.run, outdir, judge)
    go("build_fact_state", fact_state.run, outdir)
    go("build_temporal_profile", temporal_profile.run, outdir)
    go("build_fact_relations", fact_relations.run, outdir, judge)
    go("build_fact_views", fact_views.run, outdir)
    if "validate" in wanted:
        print("-- validate ...", flush=True)
        t0 = time.monotonic()
        assert_stage_inputs("validate", outdir)
        validation_result = validation.run(outdir)
        elapsed = time.monotonic() - t0
        result["stages"]["validate"] = {"success": validation_result["success"],
                                        "elapsed_s": round(elapsed, 1)}
        result["validation"] = validation_result
        assert_stage_outputs("validate", outdir)
        print(f"   validate: success={validation_result['success']} "
              f"({fmt_secs(elapsed)})", flush=True)

    # LLM accounting: the Judge's own counters plus the time layer's separate
    # LLM stats (time_extraction drives the model directly, not via Judge).
    result["llm"] = judge.report()
    time_llm = (result["stages"].get("extract_time", {}).get("llm") or {})
    time_calls = sum(int(v.get("calls", 0)) for v in time_llm.values())
    time_hits = sum(int(v.get("hits", 0)) for v in time_llm.values())
    result["llm"]["time_extraction"] = time_llm
    result["llm"]["_total_calls_including_time"] = (
        int(result["llm"].get("_total_calls", 0)) + time_calls
    )
    result["llm"]["_total_cache_hits_including_time"] = (
        int(result["llm"].get("_total_cache_hits", 0)) + time_hits
    )
    print(
        "LLM: "
        f"{result['llm'].get('_total_calls_including_time', 0)} calls, "
        f"{result['llm'].get('_total_cache_hits_including_time', 0)} cache hits, "
        f"~{result['llm'].get('_est_cost', 0)} {result['llm'].get('_currency', 'USD')} "
        f"(list price; Batch Chat traffic bills at 50%)"
    )
    judge.close()
    return result


def _stale_stage_report(outdir: str, selected: tuple) -> dict:
    """Detect mixed-state run directories: for every stage manifest whose
    stage is NOT selected this invocation, compare the input hashes it
    recorded against the files on disk now.  A mismatch means that stage's
    outputs are STALE relative to the current directory state (the classic
    multi-invocation dir: report.json describes the last command while older
    downstream artifacts silently describe an earlier state).  Loud warning +
    recorded in report.json; not fatal, because re-running the stage is the
    legitimate fix and this check is what makes the state visible."""
    stale: dict = {}
    mdir = os.path.join(outdir, "manifests")
    if not os.path.isdir(mdir):
        return stale
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".json"):
            continue
        stage = fn[:-5]
        if stage in selected:
            continue
        try:
            with open(os.path.join(mdir, fn), encoding="utf-8") as fh:
                man = json.load(fh)
        except Exception:
            continue
        mismatched = []
        for name, sha in (man.get("inputs") or {}).items():
            cur = file_sha(os.path.join(outdir, name))
            if sha is not None and cur is not None and cur != sha:
                mismatched.append(name)
        if mismatched:
            stale[stage] = mismatched
    if stale:
        print("WARNING: stale artifacts in this run directory -- these stages'"
              " recorded inputs no longer match the files on disk (their"
              " outputs describe an EARLIER state; rerun them or discard):")
        for stage, names in stale.items():
            print(f"    {stage}: {', '.join(names)}")
    return stale


def run(options: RunOptions) -> dict:
    outdir = os.path.abspath(options.outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    options = RunOptions(**{**asdict(options), "outdir": outdir})
    selected = select_stages(
        from_stage=options.from_stage,
        to_stage=options.to_stage,
    )

    chunks_path = options.chunks_path
    if not chunks_path and options.set_id:
        chunks_path = PC.chunks_path(options.set_id)
    if _needs_chunks(selected) and not chunks_path:
        raise ValueError("screen/extract/audit require --chunks or --set")
    if chunks_path:
        chunks_path = os.path.abspath(chunks_path)
        # The chunk corpora are the licensed filings text this pipeline runs
        # over and are not distributed with the repository, so a missing file
        # here is the normal first-run state. Fail now with the path that was
        # resolved, rather than deep inside read_jsonl on the first stage.
        if _needs_chunks(selected) and not os.path.isfile(chunks_path):
            raise FileNotFoundError(
                f"chunk set not found: {chunks_path}\n"
                f"  set: {options.set_id or '(from --chunks)'}\n"
                f"  the chunk corpora are licensed filings text and are not\n"
                f"  included in this repository. Point --chunks at your own\n"
                f"  .jsonl, or set AICAP_DATA_DIR to the directory holding it."
            )

    report: dict = {
        "pipeline": "aicap",
        "schema_version": SCHEMA_VERSION,
        "run_tag": PC.RUN_TAG,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_stages": list(selected),
        "options": dict(asdict(options)),
        "stages": {},
        # Mixed-state detection: stages whose manifests recorded different
        # input hashes than the files now on disk (their outputs are stale).
        "stale_stages": _stale_stage_report(outdir, selected),
    }

    for name in selected:
        if name in ("screen", "extract", "audit"):
            assert chunks_path is not None
            t0 = time.monotonic()
            assert_stage_inputs(name, outdir)
            stage_stats = _run_extraction_stage(name, options, chunks_path)
            assert_stage_outputs(name, outdir)
            elapsed = time.monotonic() - t0
            if isinstance(stage_stats, dict):
                stage_stats.setdefault("elapsed_s", round(elapsed, 1))
            report["stages"][name] = stage_stats
            print(f"   {name}: done in {fmt_secs(elapsed)}", flush=True)

    registry_stages = tuple(
        name for name in selected if name in REGISTRY_STAGE_NAMES
    )
    # DEC-027 release boundary, enforced HERE so the CLI and the Python API
    # share one definition: the deferred stages (Fact relations, quarterly
    # views) run only when explicitly requested -- never because the default
    # stage range happens to span them.  The default run therefore executes
    # every Fact-level stage THROUGH validate (whose relation/view checks
    # skip when those artifacts are absent).
    views_wanted = (options.views
                    or options.to_stage == "build_fact_views"
                    or options.from_stage == "build_fact_views")
    relations_wanted = (options.relations
                        or options.to_stage == "build_fact_relations"
                        or options.from_stage == "build_fact_relations")
    if not views_wanted:
        registry_stages = tuple(n for n in registry_stages
                                if n != "build_fact_views")
    if not relations_wanted:
        registry_stages = tuple(n for n in registry_stages
                                if n != "build_fact_relations")
    if registry_stages:
        source_dir = os.path.abspath(options.source_dir or outdir)
        registry_report = run_registry(
            indir=source_dir,
            outdir=outdir,
            stages=registry_stages,
            mock=options.mock,
            batch=options.batch,
        )
        report["stages"].update(registry_report.get("stages", {}))
        report["llm"] = registry_report.get("llm", {})
        report["validation"] = registry_report.get("validation")

    print_run_table(report["stages"], report.get("llm") or {})
    write_json(report, os.path.join(outdir, "report.json"))
    # report.json is overwritten wholesale per invocation; the append-only
    # log preserves what every earlier invocation into this directory ran,
    # so a multi-invocation dir keeps a complete provenance record.
    log_line = {
        "ts": report["started_at"],
        "selected_stages": report["selected_stages"],
        "options": report["options"],
        "stale_stages": report["stale_stages"],
        "validation_success": (report.get("validation") or {}).get("success"),
    }
    with open(os.path.join(outdir, "reports_log.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(log_line, ensure_ascii=False, default=str) + "\n")
    return report
