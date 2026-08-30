#!/usr/bin/env python3
"""Stage `extract_time` -- observation-level time claims (gate/read/check/resolve).

Drives aicap.time_normalizer over every observation:
  * the fiscal entity for each row is the row's OWN resolved target canonical
    (first member), with the chunk filer as fallback;
  * READ calls are batch-prefilled through the Message Batches API; RESOLVE
    calls (a handful per hundred rows) are synchronous and cached;
  * outputs one row per claim (obs_time_claims.jsonl) and one flag row per
    observation (obs_time_flags.jsonl).  `time_needs_review` is informational
    (2026-08-19 ruling: time signals never block an observation) -- it feeds
    claim demotion in the temporal profile and an info flag at the gate.

Cache warm-up: before calling, the source run's time/escalation cache rows
are copied in (INSERT OR IGNORE) from <indir>/cache.sqlite.  Byte-identical
requests therefore reuse the source run's results and cost nothing; only
changed rows re-bill.  NOTE: the warm-up is NOT gated on --mock -- a mock run
with a warmed cache replays real cached answers where they exist and canned
empty answers where they do not.  Treat mock output as wiring-test output,
never as a model-free baseline.
"""
from __future__ import annotations

import os
import sqlite3

from ..llm import LLM, ResultCache
from ..time_normalizer import TimeNormalizer, make_mock_fn
from .common import Manifest, read_jsonl, write_jsonl


def _warm_cache_from(indir: str, outdir: str) -> int:
    """Copy the source run's s4n_time / s4n_esc cache rows into this run's
    cache (content-addressed keys; INSERT OR IGNORE).

    Column-explicit on purpose: `INSERT ... SELECT *` breaks the moment the
    source and destination caches disagree on column count (a v1 cache has 10
    columns, the v2 schema 13), and the old fallback CREATE TABLE here even
    encoded the STALE 10-column shape.  The destination is created through
    ResultCache (the one authoritative DDL) and only the columns present on
    BOTH sides are copied."""
    src = os.path.join(indir, "cache.sqlite")
    if not os.path.exists(src):
        return 0
    ResultCache(os.path.join(outdir, "cache.sqlite")).close()   # current DDL
    dst = sqlite3.connect(os.path.join(outdir, "cache.sqlite"))
    dst.execute("ATTACH DATABASE ? AS base", (src,))
    dst_cols = [r[1] for r in dst.execute("PRAGMA table_info(results)")]
    src_cols = {r[1] for r in dst.execute("PRAGMA base.table_info(results)")}
    shared = [c for c in dst_cols if c in src_cols]
    collist = ",".join(shared)
    cur = dst.execute(
        f"INSERT OR IGNORE INTO results({collist}) "
        f"SELECT {collist} FROM base.results "
        "WHERE stage IN ('s4n_time','s4n_esc') AND status='ok'")
    n = cur.rowcount
    dst.commit()
    dst.execute("DETACH DATABASE base")
    dst.close()
    return n


def run(outdir: str, indir: str | None = None, mock: bool = False,
        batch: bool = True) -> dict:
    man = Manifest(outdir, "extract_time")
    obs_path = os.path.join(outdir, "observations.jsonl")
    res_path = os.path.join(outdir, "observations_resolved.jsonl")
    man.add_input(obs_path)
    man.add_input(res_path)
    rows = read_jsonl(obs_path)
    resolved = {r["raw_id"]: r for r in read_jsonl(res_path)}

    warmed = _warm_cache_from(indir, outdir) if indir else 0

    llm = LLM(mock=mock, mock_fn=make_mock_fn() if mock else None)
    cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
    norm = TimeNormalizer(llm, cache, mock=mock)

    def fiscal_entity_of(rid: str):
        canons = [c for c in (resolved.get(rid, {}).get("target_canonicals") or []) if c]
        return canons[0] if canons else None

    pairs = [(row, fiscal_entity_of(row["raw_id"])) for row in rows]
    # The READ prefill is a batch cache-warmer; with batch=False it is skipped
    # and each row's READ call happens synchronously inside normalize_time.
    if batch:
        norm.prefill_time_batch(pairs, label="time")

    flags_rows, claim_rows = [], []
    for row, fe in pairs:
        review, events = norm.normalize_time(row, fe)
        for e in events:
            rec = dict(e)
            rec["claim_id"] = rec.pop("temporal_event_id")
            claim_rows.append(rec)
        flags_rows.append({"raw_id": row["raw_id"],
                           "time_needs_review": review,
                           "n_claims": len(events)})

    path = os.path.join(outdir, "obs_time_claims.jsonl")
    write_jsonl(claim_rows, path)
    flag_path = os.path.join(outdir, "obs_time_flags.jsonl")
    write_jsonl(flags_rows, flag_path)
    man.add_output(path, len(claim_rows))
    man.add_output(flag_path, len(flags_rows))

    stats = {"rows": len(rows), "claims": len(claim_rows),
             "cache_rows_warmed_from_source": warmed,
             "time_tally": dict(norm.time_tally),
             "llm": {k: dict(v) for k, v in norm.llm_stats.items()
                     if k in ("time", "esc")}}
    man.done(stats)
    cache.close()
    return stats
