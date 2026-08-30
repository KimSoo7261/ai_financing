#!/usr/bin/env python3
"""Extraction: chunks -> screened -> extracted -> audited capital instances.

    chunks.jsonl
        |  run_stage1 (screen, Haiku)   -- is an AI-infra capital obligation
        |     present?  Plus a free deterministic pre-skip of chunks with no
        |     obligation keyword, no $ figure, and no capacity cue.
        v
    relevant chunks (flags live in the cache)
        |  run_stage2 (extract + classify, Opus) -- pull each atomic capital
        |     observation with its verbatim quote and classification axes
        |     -> raw_stage2.jsonl
        v
    raw observations
        |  classify + GATE (aicap.assemble) -- keep iff target + placeable
        |     magnitude; failures -> dropped.jsonl with a drop_reason
        v
    kept rows
        |  run_stage3 (review, Opus) -- PURE AUDITOR: verify every extracted
        |     field, grade eight checks, reconstruct the expected roles; the
        |     admission policy below turns grades + evidence into
        |     admit / drop / human-direction-lane routing
        v
    outdir/  capital_instances.csv/.jsonl, audit.csv/.jsonl, raw_stage2.jsonl,
             dropped.jsonl, direction_review_queue.jsonl/.csv,
             drop_spotcheck.jsonl, run_stats.json

Everything downstream of Stage 3 (amount/entity/time normalization, Fact
registry, quarterly views) lives in aicap/stages and is sequenced by
aicap.orchestrator.

Resumability: results cache in outdir/cache.sqlite keyed on
(id, stage, prompt_version, model_id).  Re-running skips completed work;
editing a prompt invalidates only that stage.  mock=True runs all stages with
a canned model (no API key, no cost).
"""
from __future__ import annotations
import os, re, json, time
from collections import Counter
from functools import lru_cache
from typing import Optional

from .llm import (LLM, CallConfig, ResultCache, read_jsonl, write_jsonl,
                  write_csv, ensure_dir, sha16)
from .config import (MODELS, SCREEN_MODEL, EXTRACT_MODEL, REVIEW_MODEL,
                     stage_models, DROP_SPOTCHECK_RATE)
from .prompts import (SCREEN_PROMPT, SCREEN_VERSION, EXTRACT_PROMPT,
                      EXTRACT_VERSION, REVIEW_PROMPT, REVIEW_VERSION,
                      build_user, build_review_user, review_content_digest,
                      valid_screen, valid_extract, valid_review, REVIEW_CHECK_KEYS,
                      REVIEW_ROLE_KEYS)
from . import assemble as D
from . import classify as V
from .escalation import (direction_escalation, emitted_role_mismatch,
                         should_mismatch, auto_direction_fix,
                         consensus_clear_counterparty,
                         inverted_anchor, degenerate_slots, degenerate_slot_fix)


# ===========================================================================
# E1/VAL-3 (2026-08-20): live cost telemetry for the EXPENSIVE stages.
# The registry Judge always metered itself, but screen/extract/review --
# $13.78 of claudetest's $17.33 -- were invisible to report.json (the 12x
# undercount, reproduced twice).  Every fresh (non-cache-hit) call funnels
# through USAGE_METER.add at the same points that increment n_called, so
# batch write-through can never double-count.  Costs are LIST price in the
# model's own currency; Batch traffic actually bills at 50%.
# ===========================================================================
_MODEL_KEY_OF_ID = {spec["id"]: key for key, spec in MODELS.items()}


class _UsageMeter:
    def __init__(self):
        self.by_stage: dict = {}

    def add(self, stage: str, model_id: str, usage: dict) -> None:
        st = self.by_stage.setdefault(stage, {
            "model_id": model_id, "calls": 0, "in_tokens": 0, "out_tokens": 0})
        st["calls"] += 1
        st["in_tokens"] += int(usage.get("in_tokens") or 0)
        st["out_tokens"] += int(usage.get("out_tokens") or 0)

    def snapshot(self, mock: bool = False) -> dict:
        out: dict = {"_mock_projection": True} if mock else {}
        totals: dict = {}
        for stage, st in self.by_stage.items():
            spec = MODELS.get(_MODEL_KEY_OF_ID.get(st["model_id"], ""), {})
            cur = spec.get("currency", "USD")
            cost = round((st["in_tokens"] * spec.get("in", 0.0)
                          + st["out_tokens"] * spec.get("out", 0.0)) / 1e6, 4)
            out[stage] = {**st, "est_cost_list": cost, "currency": cur}
            totals[cur] = round(totals.get(cur, 0.0) + cost, 4)
        out["_total_est_cost_list_by_currency"] = totals
        return out


USAGE_METER = _UsageMeter()


# ===========================================================================
# Stage 1 -- relevance screen
# ===========================================================================
SCREEN_STAGE = "screen"

# Seed payloads written by --no-screen carry this marker so a later REAL
# screen run re-screens them instead of accepting a fabricated verdict as a
# final screen result (the seeds sit under the same cache key as real
# verdicts and previously masqueraded as screened output).
PRESEED_REASON = "pre-screened (stage 1 skipped)"


def _chunk_cache_id(chunk: dict) -> str:
    """Cache identity of one chunk for the screen/extract stages:
    chunk_id + a digest of the chunk TEXT.  chunk_id alone is positional
    (sha1(doc_id, section, index)), so a corpus refresh that edits text under
    a stable chunk_id would silently replay stale cached verdicts; folding
    the text digest in makes any text change a fresh cache identity (the
    Stage-3 review key has used this content-digest pattern all along)."""
    return f"{chunk['chunk_id']}|txt:{sha16(chunk.get('text') or '')}"


# Capacity-cue backstop for the free pre-skip.  The chunker's
# candidate_categories already carry MW/GW/PPA cues -- this guards the
# residual gap where a filing states compute/power capacity with no category
# keyword and no $ figure.  Widening only ever SENDS MORE chunks to the
# Stage-1 screen (recall-first); it can never drop one.
_CAPACITY_RE = re.compile(
    r"\b(?:MW|GW|megawatts?|gigawatts?|GPUs?|TPUs?|xPUs?|accelerators?|"
    r"H[12]00s?|GB[23]00s?|exa[fF][lL][oO][pP][sS]?)\b")


def should_preskip(chunk: dict) -> bool:
    """Free deterministic skip: no obligation keyword AND no $ figure AND no
    capacity cue in the text."""
    if chunk.get("candidate_categories") or chunk.get("has_dollar"):
        return False
    return not _CAPACITY_RE.search(chunk.get("text") or "")


def run_screen(chunks_path: str, cache: ResultCache, llm: LLM,
               model_key: str = SCREEN_MODEL, preskip: bool = True,
               limit: Optional[int] = None, batch: bool = True) -> dict:
    cfg = CallConfig(model_key=model_key, max_tokens=300)
    model_id = cfg.model_id

    # Batch-prefill the cache misses (half token cost; thousands of chunks
    # would take hours sequentially).  Earlier-run quarantines enter the
    # prefill too -- that IS their retry for this run; whatever they come
    # back as is final (no synchronous double-billing below).
    n_called = n_quar = 0
    run_final: set = set()
    if batch and not llm.mock:
        reqs = []
        n_seen = 0
        for chunk in read_jsonl(chunks_path):
            if limit and n_seen >= limit:
                break
            n_seen += 1
            if preskip and should_preskip(chunk):
                continue
            cid = chunk["chunk_id"]
            key = cache.make_key(_chunk_cache_id(chunk), SCREEN_STAGE,
                                 SCREEN_VERSION, model_id)
            cc = cache.get(key)
            # A --no-screen seed is NOT a screen verdict: re-screen it.
            if cc is not None and cc.get("status") == "ok" \
                    and not (cc.get("payload") or {}).get("preseeded"):
                continue
            reqs.append(((cid, key), SCREEN_PROMPT, build_user(chunk)))
        if reqs:
            def _persist(ck, res):
                cid_, key_ = ck
                obj_, usage_, status_ = res
                cache.put(key_, cid_, SCREEN_STAGE, SCREEN_VERSION, model_id,
                          status_, obj_,
                          usage_["in_tokens"], usage_["out_tokens"],
                          stop_reason=usage_.get("stop_reason"),
                          failure_reason=usage_.get("failure_reason"),
                          telemetry=usage_)
            results = llm.call_json_batch(reqs, cfg, validate=valid_screen,
                                          label="screen", on_result=_persist)
            for (cid, key), (obj, usage, status) in results.items():
                cache.put(key, cid, SCREEN_STAGE, SCREEN_VERSION, model_id,
                          status, obj, usage["in_tokens"], usage["out_tokens"],
                          stop_reason=usage.get("stop_reason"),
                          failure_reason=usage.get("failure_reason"), telemetry=usage)
                n_called += 1
                USAGE_METER.add(SCREEN_STAGE, model_id, usage)
                if status != "ok":
                    n_quar += 1
                run_final.add(key)

    n_seen = n_relevant = n_skip = 0
    for chunk in read_jsonl(chunks_path):
        if limit and n_seen >= limit:
            break
        n_seen += 1
        cid = chunk["chunk_id"]
        if preskip and should_preskip(chunk):
            n_skip += 1; continue
        key = cache.make_key(_chunk_cache_id(chunk), SCREEN_STAGE,
                             SCREEN_VERSION, model_id)
        cached = cache.get(key)
        # An OK result is final (a --no-screen seed is not a result: it is
        # re-screened); a quarantine this run already retried (via the
        # prefill) is final for the run; an earlier-run quarantine on the
        # non-batch path retries synchronously.
        if cached is not None and cached["status"] == "ok" \
                and not (cached.get("payload") or {}).get("preseeded"):
            if cached["payload"].get("relevant") in (1, True):
                n_relevant += 1
            continue
        if cached is not None and key in run_final:
            continue
        obj, usage, status = llm.call_json(SCREEN_PROMPT, build_user(chunk), cfg, validate=valid_screen)
        cache.put(key, cid, SCREEN_STAGE, SCREEN_VERSION, model_id, status, obj,
                  usage["in_tokens"], usage["out_tokens"],
                  stop_reason=usage.get("stop_reason"),
                  failure_reason=usage.get("failure_reason"), telemetry=usage)
        n_called += 1
        USAGE_METER.add(SCREEN_STAGE, model_id, usage)
        if status == "quarantine":
            n_quar += 1
        elif obj.get("relevant") in (1, True):
            n_relevant += 1
        if n_called % 200 == 0:
            print(f"  screen: seen {n_seen}, called {n_called}, relevant {n_relevant}, quarantined {n_quar}")
    return {"seen": n_seen, "called": n_called, "relevant": n_relevant,
            "preskipped": n_skip, "quarantined": n_quar,
            "batched": bool(batch and not llm.mock),
            "prompt_version": SCREEN_VERSION,
            "llm_usage": USAGE_METER.snapshot(mock=llm.mock)}


def relevant_chunk_ids(cache: ResultCache, chunks_by_id: dict,
                       screen_model: str = SCREEN_MODEL) -> set:
    """Chunk ids of THIS corpus whose cached screen verdict (current screen
    prompt version, current screen model, current chunk TEXT) says relevant.

    Probes the cache per current chunk instead of scanning every screen row:
    the old scan read ALL status='ok' screen rows regardless of prompt
    version, model, or corpus membership, so a reused outdir unioned stale
    corpora and superseded screen-prompt verdicts into the relevant set."""
    model_id = CallConfig(model_key=screen_model).model_id
    out = set()
    for cid, chunk in chunks_by_id.items():
        key = cache.make_key(_chunk_cache_id(chunk), SCREEN_STAGE,
                             SCREEN_VERSION, model_id)
        cc = cache.get(key)
        if cc is not None and cc.get("status") == "ok" \
                and (cc.get("payload") or {}).get("relevant") in (1, True):
            out.add(cid)
    return out


def select_chunks(chunks_by_id: dict, relevant_ids: set,
                  limit: Optional[int] = None) -> list[str]:
    """The deterministic work list for Stage 2: sorted relevant ids that
    exist in the CURRENT chunks file, capped at `limit` AFTER the membership
    check (a stale id must never consume this run's quota)."""
    out = []
    for cid in sorted(relevant_ids):
        if cid not in chunks_by_id:
            continue
        out.append(cid)
        if limit and len(out) >= limit:
            break
    return out


# ===========================================================================
# Stage 2 -- capital-fact extraction + classification
# ===========================================================================
EXTRACT_STAGE = "extract"
EXTRACT_SPLIT_STAGE = "extract_split"
EXTRACT_SPLIT_VERSION = EXTRACT_VERSION + "_split_v1"


def _split_chunk_text(text: str, overlap: int = 600) -> list[str]:
    """Deterministic two-way split on a nearby paragraph/sentence boundary.

    A small overlap prevents a fact at the cut from losing its amount or role
    context. Exact/core-field de-duplication after recovery removes repeated
    observations from the overlap.
    """
    text = str(text or "")
    if len(text) < 2:
        return [text]
    mid = len(text) // 2
    probes = [text.rfind("\n\n", 0, mid), text.find("\n\n", mid),
              text.rfind(". ", 0, mid), text.find(". ", mid)]
    cuts = [p + (2 if text[p:p + 2] in {"\n\n", ". "} else 0)
            for p in probes if p >= 0]
    cut = min(cuts, key=lambda p: abs(p - mid)) if cuts else mid
    cut = max(1, min(len(text) - 1, cut))
    return [text[:min(len(text), cut + overlap)], text[max(0, cut - overlap):]]


def _dedupe_recovered_observations(rows: list[dict]) -> list[dict]:
    """Collapse overlap duplicates without fuzzy-merging different facts."""
    seen, out = set(), []
    for raw in rows:
        o = {k: v for k, v in raw.items() if not k.startswith("_truncated_")}
        key = tuple(json.dumps(o.get(k), ensure_ascii=False, sort_keys=True)
                    for k in ("verbatim_quote", "target_firm", "counterparty",
                              "amount_text", "quantity_value", "quantity_unit",
                              "instrument_class", "amount_basis",
                              "amount_aggregation_level", "object_name"))
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out


def _recover_partial_extraction(chunk: dict, cache: ResultCache, llm: LLM,
                                cfg: CallConfig, model_id: str,
                                *, depth: int = 0, max_depth: int = 4) -> tuple:
    """Split and re-extract a provider-confirmed partial response.

    Returns (complete_payload|None, accumulated_usage). A parent can become
    cache-status `ok` only when every leaf completed under the strict schema.
    """
    usage_total = {"in_tokens": 0, "out_tokens": 0, "split_calls": 0,
                   "stop_reason": "end_turn"}
    text = str(chunk.get("text") or "")
    if depth >= max_depth or len(text) < 800:
        usage_total["failure_reason"] = "partial_split_exhausted"
        return None, usage_total
    recovered: list[dict] = []
    for index, part in enumerate(_split_chunk_text(text)):
        child = dict(chunk)
        child["text"] = part
        child_id = f"{chunk.get('chunk_id')}::split:{depth + 1}:{index}:{sha16(part)}"
        key = cache.make_key(child_id, EXTRACT_SPLIT_STAGE,
                             EXTRACT_SPLIT_VERSION, model_id)
        hit = cache.get(key)
        if hit is not None and hit.get("status") == "ok":
            obj, status = hit["payload"], "ok"
            child_usage = {"in_tokens": 0, "out_tokens": 0,
                           "stop_reason": hit.get("stop_reason") or "cache_hit"}
        else:
            obj, child_usage, status = llm.call_json(
                EXTRACT_PROMPT, build_user(child), cfg, validate=valid_extract)
            cache.put(key, child_id, EXTRACT_SPLIT_STAGE, EXTRACT_SPLIT_VERSION,
                      model_id, status, obj, child_usage.get("in_tokens", 0),
                      child_usage.get("out_tokens", 0),
                      stop_reason=child_usage.get("stop_reason"),
                      failure_reason=child_usage.get("failure_reason"),
                      telemetry=child_usage)
        usage_total["in_tokens"] += child_usage.get("in_tokens", 0)
        usage_total["out_tokens"] += child_usage.get("out_tokens", 0)
        usage_total["split_calls"] += 1 + child_usage.get("split_calls", 0)
        if status == "partial":
            nested, nested_usage = _recover_partial_extraction(
                child, cache, llm, cfg, model_id, depth=depth + 1,
                max_depth=max_depth)
            usage_total["in_tokens"] += nested_usage.get("in_tokens", 0)
            usage_total["out_tokens"] += nested_usage.get("out_tokens", 0)
            usage_total["split_calls"] += nested_usage.get("split_calls", 0)
            if nested is None:
                usage_total["failure_reason"] = nested_usage.get(
                    "failure_reason", "partial_split_failed")
                return None, usage_total
            recovered.extend(nested["observations"])
        elif status == "ok":
            recovered.extend(obj["observations"])
        else:
            usage_total["failure_reason"] = child_usage.get(
                "failure_reason", "split_leaf_quarantine")
            return None, usage_total
    return {"observations": _dedupe_recovered_observations(recovered)}, usage_total


def _enrich(inst: dict, chunk: dict) -> dict:
    """Attach immutable provenance so each instance is auditable."""
    inst = dict(inst)
    for k in ("firm", "source_type", "fiscal_period", "doc_date", "section_label",
              "chunk_id", "doc_id", "char_start", "char_end", "source_path", "firms_mentioned"):
        inst[k] = chunk.get(k)
    return inst


# ===========================================================================
# Big-amount rescue for STRUCTURALLY failed extractions (2026-08-28).
# When parse + one repair both fail and the payload is quarantined, a payload
# whose extracted amounts reach RESCUE_MIN_USD gets exactly ONE more LLM
# round: the FAILED RAW TEXT is sent back with a fix-JSON-only instruction
# (never the full chunk -- the observed real quarantines sat on 17k-24k-token
# chunks).  The rescue result is cached under its own namespace with ANY
# status, so the attempt is structurally one-shot per (chunk text, extract
# prompt, model) ACROSS runs -- it can never loop.
# Empirical note: the deterministic lenient parse in llm.extract_json already
# recovers every structural failure observed on the real 2025 runs (all were
# control-character JSON breaks with sub-$10B amounts), so this lane is
# insurance for the not-yet-observed big-amount case.
# ===========================================================================
RESCUE_STAGE = "extract_rescue"
RESCUE_VERSION = EXTRACT_VERSION + "_rescue_v1"
RESCUE_MIN_USD = float(os.environ.get("AICAP_RESCUE_MIN_USD", str(10e9)))

_PROSE_BIG_RE = re.compile(
    r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(trillion|billion)\b", re.I)
_FIELD_PAIR_RE = re.compile(
    r'"amount_text"\s*:\s*"([^"]{1,80})"[^{}]{0,400}?"amount_scale"\s*:\s*"([^"]{1,20})"',
    re.S)


def _payload_max_usd(raw_text) -> Optional[float]:
    """Largest USD amount recoverable from a failed extraction payload.

    Tiered, deterministic, zero-cost: (1) lenient-parse the whole payload;
    (2) brace-walk complete observation objects out of it; (3) regex the
    amount_text/amount_scale field pairs plus prose '$N billion/trillion'.
    Amounts are read through classify_amount so '$6,529' + scale 'million'
    counts as $6.5B -- a '$NN billion' regex alone misses exactly the format
    real failures used.  Returns None when nothing monetary is found."""
    if not isinstance(raw_text, str) or not raw_text.strip() \
            or raw_text.startswith("TRANSPORT_ERROR"):
        return None
    best = None

    def _see(amount_text, amount_scale):
        nonlocal best
        try:
            bn = V.classify_amount(amount_text, amount_scale).get("amount_bn")
        except Exception:
            return
        if bn is not None:
            usd = float(bn) * 1e9
            best = usd if best is None or usd > best else best

    rows = []
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "",
                                raw_text.strip(), flags=re.S), strict=False)
        rows = obj.get("observations") or [] if isinstance(obj, dict) else []
    except Exception:
        from .llm import salvage_observations
        rows = salvage_observations(raw_text)
    for o in rows:
        if isinstance(o, dict):
            _see(o.get("amount_text"), o.get("amount_scale"))
    if best is None:
        for m in _FIELD_PAIR_RE.finditer(raw_text):
            _see(m.group(1), m.group(2))
        for m in _PROSE_BIG_RE.finditer(raw_text):
            _see(m.group(0), None)
    return best


def _rescue_quarantined_extraction(chunk: dict, raw_text: str,
                                   cache: ResultCache, llm: LLM,
                                   cfg: CallConfig, model_id: str) -> tuple:
    """One-shot repair of a quarantined big-amount payload.

    Returns (payload | None, usage).  The rescue key stores ANY outcome, so
    a chunk is rescued at most once per (text, prompt, model) across runs."""
    key = cache.make_key(_chunk_cache_id(chunk), RESCUE_STAGE, RESCUE_VERSION,
                         model_id)
    hit = cache.get(key)
    if hit is not None:
        payload = hit["payload"] if hit.get("status") == "ok" else None
        return payload, {"in_tokens": 0, "out_tokens": 0,
                         "stop_reason": "cache_hit"}
    fix_user = (
        "A previous response to this extraction task was structurally invalid "
        "JSON. Repair it into valid JSON that matches the required output "
        "schema EXACTLY. Fix JSON syntax and schema shape ONLY -- do NOT "
        "re-extract, add, drop, or alter any extracted value or quote.\n\n"
        "INVALID RESPONSE:\n" + str(raw_text)[:80000] +
        "\n\nRespond with ONLY the corrected JSON, no prose, no code fences.")
    obj, usage, status = llm.call_json(EXTRACT_PROMPT, fix_user, cfg,
                                       validate=valid_extract)
    usage = dict(usage)
    usage["rescue_attempted"] = True
    cache.put(key, chunk["chunk_id"], RESCUE_STAGE, RESCUE_VERSION, model_id,
              status, obj, usage.get("in_tokens", 0), usage.get("out_tokens", 0),
              stop_reason=usage.get("stop_reason"),
              failure_reason=usage.get("failure_reason"), telemetry=usage)
    return (obj if status == "ok" else None), usage


def run_extract(chunks_by_id: dict, relevant_ids: set, cache: ResultCache,
                llm: LLM, model_key: str = EXTRACT_MODEL,
                limit: Optional[int] = None, batch: bool = True) -> dict:
    # max_tokens=None -> the model maximum (a fixed cap silently truncates
    # dense chunks); stop_reason/salvage are counted below as the backstop.
    cfg = CallConfig(model_key=model_key, max_tokens=None)
    model_id = cfg.model_id
    n_called = n_inst = n_quar = n_partial = 0
    n_truncated = n_salvaged_rows = n_rescued = n_rescue_failed = 0
    truncated_chunks = []
    quarantined_chunks = []

    selected = select_chunks(chunks_by_id, relevant_ids, limit)
    pending = []                       # (chunk_id, cache_key) with no cached result
    for cid in selected:
        chunk = chunks_by_id[cid]
        key = cache.make_key(_chunk_cache_id(chunk), EXTRACT_STAGE,
                             EXTRACT_VERSION, model_id)
        cc = cache.get(key)
        # Earlier-run quarantines retry; an OK result is final.
        if cc is not None and cc.get("status") == "ok":
            continue
        pending.append((cid, key))

    # Message Batches API for the cache misses (half token cost); per-item
    # failures fall back to the synchronous path inside call_json_batch.
    results = {}
    if batch and pending and not llm.mock:
        reqs = [(key, EXTRACT_PROMPT, build_user(chunks_by_id[cid]))
                for cid, key in pending]
        cid_of = {key: cid for cid, key in pending}

        def _persist(key, res):
            # Write-through per completed item so an interrupted wave keeps
            # its paid-for results.  Enrichment must happen HERE too: the
            # cached payload is what collect_raw_instances later reads, and
            # it must carry provenance whether or not the loop below (which
            # rewrites the same rows, idempotently) ever ran.
            obj, usage, status = res
            if status == "ok":
                chunk = chunks_by_id[cid_of[key]]
                obj = {"observations": [_enrich(o, chunk)
                                        for o in obj["observations"]]}
            cache.put(key, cid_of[key], EXTRACT_STAGE, EXTRACT_VERSION,
                      model_id, status, obj,
                      usage["in_tokens"], usage["out_tokens"],
                      stop_reason=usage.get("stop_reason"),
                      failure_reason=usage.get("failure_reason"),
                      telemetry=usage)

        results = llm.call_json_batch(reqs, cfg, validate=valid_extract,
                                      label="extract", on_result=_persist)

    for cid, key in pending:
        chunk = chunks_by_id[cid]
        if key in results:
            obj, usage, status = results[key]
        else:
            obj, usage, status = llm.call_json(EXTRACT_PROMPT, build_user(chunk),
                                               cfg, validate=valid_extract)
        if usage.get("stop_reason") == "max_tokens":
            n_truncated += 1
            n_salvaged_rows += usage.get("salvaged", 0)
            truncated_chunks.append(cid)
        if status == "partial":
            recovered, split_usage = _recover_partial_extraction(
                chunk, cache, llm, cfg, model_id)
            usage["in_tokens"] += split_usage.get("in_tokens", 0)
            usage["out_tokens"] += split_usage.get("out_tokens", 0)
            usage["split_calls"] = split_usage.get("split_calls", 0)
            if recovered is not None:
                obj, status = recovered, "ok"
                usage["recovered_by_split"] = True
                usage["failure_reason"] = None
                print(f"  extract: recovered max-token partial chunk {cid} "
                      f"with {usage['split_calls']} split call(s)")
            else:
                n_partial += 1
                usage["failure_reason"] = split_usage.get(
                    "failure_reason", "partial_split_failed")
                print(f"  extract WARNING: chunk {cid} remains PARTIAL after "
                      f"{usage.get('split_calls', 0)} split call(s); no complete "
                      f"parent success will be cached")
        if status == "quarantine":
            # Big-amount one-shot rescue: only for structural (parse/schema)
            # failures whose payload text carries an amount >= RESCUE_MIN_USD.
            big = _payload_max_usd(obj if isinstance(obj, str) else None)
            if big is not None and big >= RESCUE_MIN_USD:
                rescued, r_usage = _rescue_quarantined_extraction(
                    chunk, obj, cache, llm, cfg, model_id)
                usage["in_tokens"] += r_usage.get("in_tokens", 0)
                usage["out_tokens"] += r_usage.get("out_tokens", 0)
                usage["rescue_attempted"] = True
                usage["rescue_payload_max_usd"] = big
                if rescued is not None:
                    obj, status = rescued, "ok"
                    usage["failure_reason"] = None
                    usage["rescued_big_amount"] = True
                    n_rescued += 1
                    print(f"  extract: RESCUED quarantined chunk {cid} "
                          f"(payload amount ~${big / 1e9:.1f}B) with one repair round")
                else:
                    n_rescue_failed += 1
        if status == "ok":
            obj = {"observations": [_enrich(o, chunk) for o in obj["observations"]]}
            n_inst += len(obj["observations"])
        elif status == "quarantine":
            n_quar += 1
            quarantined_chunks.append(cid)
        cache.put(key, cid, EXTRACT_STAGE, EXTRACT_VERSION, model_id, status, obj,
                  usage["in_tokens"], usage["out_tokens"],
                  stop_reason=usage.get("stop_reason"),
                  failure_reason=usage.get("failure_reason"), telemetry=usage)
        n_called += 1
        USAGE_METER.add(EXTRACT_STAGE, model_id, usage)
        if n_called % 100 == 0:
            print(f"  extract: called {n_called}, instances {n_inst}, quarantined {n_quar}")
    return {"called": n_called, "instances": n_inst, "quarantined": n_quar,
            "partial_unresolved": n_partial,
            "quarantined_chunks": quarantined_chunks,
            "rescued_big_amount": n_rescued,
            "rescue_attempted_failed": n_rescue_failed,
            "rescue_min_usd": RESCUE_MIN_USD,
            "truncated_chunks": truncated_chunks, "truncation_salvaged_rows": n_salvaged_rows,
            "batched": bool(batch and not llm.mock),
            "prompt_version": EXTRACT_VERSION,
            "llm_usage": USAGE_METER.snapshot(mock=llm.mock)}


def collect_raw_instances(cache: ResultCache, chunks_by_id: dict,
                          selected: list[str], model_id: str) -> list:
    """The ORIGINAL Stage-2 observations (enriched with provenance), each given
    a stable raw_id.  This is the preserved raw -- Stage 3 operates on it.

    Scoped to THIS run: only the selected chunk ids (current corpus, current
    relevance set, current --limit), probed under the current extract prompt
    version, extract model, and chunk TEXT digest.  The old collector took
    every ok payload for the prompt version, so a reused outdir leaked a
    prior larger run past a shrunk --limit, a different corpus past a
    --set/--chunks switch, and a different extract model's rows into
    raw_stage2.jsonl.  A cached ok row for a selected chunk still replays
    free (resumability is chunk-scoped, not scan-scoped).

    raw_id = sha16(chunk_id, in-chunk ordinal, quote[:60]).  The ordinal is
    the observation's position in the chunk's cached extraction output, so it
    is stable across reruns and unique by construction (two sibling
    observations in one chunk can share a quote prefix; the ordinal keeps
    their ids distinct)."""
    out = []
    for cid in selected:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        key = cache.make_key(_chunk_cache_id(chunk), EXTRACT_STAGE,
                             EXTRACT_VERSION, model_id)
        cc = cache.get(key)
        if cc is None or cc.get("status") != "ok":
            continue
        try:
            for i, o in enumerate((cc["payload"] or {}).get("observations", [])):
                o["raw_id"] = sha16(o.get("chunk_id"), f"{i:03d}",
                                    (o.get("verbatim_quote") or "")[:60])
                out.append(o)
        except Exception:
            pass
    return out


# ===========================================================================
# Stage 3 -- PURE defensive review (audit only)
# ===========================================================================
REVIEW_STAGE = "review"

_WS_RE = re.compile(r"\s+")
# Normalize away two classes of FALSE verbatim failures:
# (a) private-use-area glyphs (slide/PDF bullet characters like
#  that extractors legitimately drop), (b) inline wire-ticker tags such as
# "<AKER.OL>" that models elide from quotes. Applied to BOTH sides of the check.
_PUA_RE = re.compile("[-]")
_TICKER_TAG_RE = re.compile(r"<[A-Z0-9][A-Z0-9._-]{0,14}>")


# (c) claudetest punch #3 (2026-08-20): apostrophe/quote variants and
# cp1252-mojibake marker characters ("Japanâs" in the wire text vs the
# model's "Japan's") are DELETED on both sides of the check, so they can
# never be the sole difference that fatals a quote.  Deleting punctuation-
# class characters symmetrically cannot hide an invented word or numeral --
# those tokens still mismatch.
_APOS_MOJI_RE = re.compile(
    "[\u0027\u0060\u00b4\u2018\u2019\u201a\u201b\u02bc"   # apostrophes
    "\u0022\u201c\u201d\u201e\u00ab\u00bb"                  # double quotes
    "\u00e2\u00c3\u20ac\u2122\u0153\u0080-\u009f]"        # mojibake markers
)


def _vnorm(text) -> str:
    s = _PUA_RE.sub(" ", str(text))
    s = _TICKER_TAG_RE.sub(" ", s)
    s = _APOS_MOJI_RE.sub("", s)
    return _WS_RE.sub(" ", s)


def _verbatim_pass(quote, excerpt) -> bool:
    """The one mechanical audit done in code: every ' [...] '-joined span of
    the quote must be a normalised substring of the excerpt (whitespace
    collapsed, PUA glyphs and inline ticker tags stripped on both sides)."""
    if not quote or not excerpt:
        return False
    ex = _vnorm(excerpt)
    for span in str(quote).split(" [...] "):
        s = _vnorm(span).strip()
        if s and s not in ex:
            return False
    return True


# GATE-1' (PI ruling 2026-08-18): the boolean above stays the frozen pass/fail
# authority; the GRADE below separates slight, non-fatal differences from
# fabrication.  Content-based, not char-similarity -- similarity mis-ranks the
# most benign class (SEC table quotes with a model-inserted "$" score 0.22).
_VB_SYMBOL_RE = re.compile(r"[$€£¥\s]")
_VB_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w.,%/-]*")
_VB_ORDER = {"pass": 0, "drift": 1, "reorder": 2, "fatal": 3}


@lru_cache(maxsize=128)
def _excerpt_forms(excerpt: str):
    """Precomputed comparison forms of one chunk text (memoised: consecutive
    rows usually share the chunk)."""
    norm = _vnorm(excerpt)
    toks = tuple(t.rstrip(".,").casefold() for t in _VB_TOKEN_RE.findall(norm))
    return norm, _VB_SYMBOL_RE.sub("", norm), toks, frozenset(toks)


def _verbatim_grade(quote, excerpt) -> tuple[str, list]:
    """(grade, missing_tokens) with grade in pass|drift|reorder|fatal:

      pass    every span a _vnorm substring (== _verbatim_pass semantics)
      drift   span matches once currency symbols/whitespace are stripped
              ("$67,201" vs bare "67,201", "(1,054.5)" vs "(1,054.5 )"),
              or its tokens appear in the excerpt IN ORDER (cell-joined
              table rows, date-prefixed timeline entries)
      reorder every token present but order broken (stitched table rows;
              kept as its own tier because for prose this is the one place
              a fabricated sentence built from real words could hide)
      fatal   >=1 quoted token absent from the excerpt -- an invented word
              or, worst, an invented numeral.  missing_tokens carries the
              evidence (capped).

    Measured on the frozen QWen run's 168 boolean fails: 114 drift /
    51 reorder / 3 fatal (two invented "$38 billion" numerals, one inserted
    "Japan")."""
    if not quote or not excerpt:
        return "fatal", []
    ex_norm, ex_squeezed, ex_toks, ex_tokset = _excerpt_forms(str(excerpt))
    worst, missing_all = "pass", []
    for span in str(quote).split(" [...] "):
        s = _vnorm(span).strip()
        if not s or s in ex_norm:
            continue
        squeezed = _VB_SYMBOL_RE.sub("", s)
        if squeezed and squeezed in ex_squeezed:
            grade = "drift"
        else:
            toks = [t.rstrip(".,") for t in _VB_TOKEN_RE.findall(s)]
            cf = [t.casefold() for t in toks]
            if not cf:
                grade = "drift"                # pure-punctuation span
            else:
                i = 0
                for t in ex_toks:              # ordered subsequence test
                    if i < len(cf) and t == cf[i]:
                        i += 1
                if i == len(cf):
                    grade = "drift"
                else:
                    missing = [toks[j] for j in range(len(cf))
                               if cf[j] not in ex_tokset]
                    if missing:
                        grade = "fatal"
                        missing_all.extend(missing)
                    else:
                        grade = "reorder"
        if _VB_ORDER[grade] > _VB_ORDER[worst]:
            worst = grade
    return worst, missing_all[:8]


# Unparsable audit -> grade every check "unsure" (route to a human); never fake a pass.
_UNPARSABLE_AUDIT = {"role_resolution": {k: ("unclear" if k == "economic_action" else None)
                                           for k in REVIEW_ROLE_KEYS},
                     "checks": {k: "unsure" for k in REVIEW_CHECK_KEYS},
                     "note": "auditor_unparsable"}


# ===========================================================================
# Admission policy + human direction-review lane
# ===========================================================================
# time_ok is NON-FATAL: the dedicated time layer downstream owns time, so a
# row whose ONLY bad check is time_ok is admitted with audit_time_bad=True
# (the time layer re-derives its claims from the quote) instead of dropped.
NONFATAL_CHECKS = {"time_ok"}
OVERRIDES_NAME = "human_overrides.jsonl"
DIRECTION_QUEUE_NAME = "direction_review_queue"
_OVERRIDE_ACTIONS = {"swap_direction", "set_roles"}


def load_overrides(outdir: str) -> dict:
    """Human direction corrections, keyed by raw_id. Each record needs raw_id,
    action in {swap_direction, set_roles} and a non-empty reviewer; set_roles
    additionally supplies target_firm / counterparty. Invalid records are
    ignored with a warning -- an override is an explicit signed artifact, never
    a guess."""
    path = os.path.join(outdir, OVERRIDES_NAME)
    overrides = {}
    if not os.path.exists(path):
        return overrides
    for rec in read_jsonl(path):
        rid = str(rec.get("raw_id") or "")
        action = rec.get("action")
        if not rid or action not in _OVERRIDE_ACTIONS or not str(rec.get("reviewer") or "").strip():
            print(f"  WARNING: ignoring invalid override record: {rec}")
            continue
        if action == "set_roles" and not (rec.get("target_firm") or rec.get("counterparty")):
            print(f"  WARNING: set_roles override without roles ignored: {rid}")
            continue
        overrides[rid] = rec
    return overrides


def apply_direction_override(row: dict, override: dict) -> dict:
    """Apply a human-approved direction correction to one observation row."""
    row = dict(row)
    if override["action"] == "swap_direction":
        row["target_firm"], row["counterparty"] = row.get("counterparty"), row.get("target_firm")
    else:                                        # set_roles
        if override.get("target_firm"):
            row["target_firm"] = override["target_firm"]
        if "counterparty" in override:
            row["counterparty"] = override.get("counterparty")
    row["direction_fix_applied"] = True
    row["direction_fix_source"] = "human_override"
    row["direction_fix_reviewer"] = override.get("reviewer")
    return row


def _direction_queue_row(row: dict, audit: dict) -> dict:
    """A self-contained direction-review item, pre-filled with Stage-3's own
    reconstructed roles as the proposed correction. The human fills `decision`
    (approve_swap | set_roles | reject) (+ corrected fields for set_roles) and
    tools/apply_direction_reviews.py converts decisions into human_overrides.jsonl.

    The row shows ALL FIVE extractor direction fields (economic_action,
    capital_provider, capital_recipient, target_firm, counterparty) AND the
    auditor's reconstruction, clearly labeled apart -- the queue once showed
    only the auditor's economic_action under an unlabeled name, so a human
    could not check the extractor's own resolution.  content_digest binds any
    resulting override to exactly this row content (see load/apply)."""
    return {
        "raw_id": row.get("raw_id"), "chunk_id": row.get("chunk_id"),
        "target_firm": row.get("target_firm"), "counterparty": row.get("counterparty"),
        "economic_action": row.get("economic_action"),
        "capital_provider": row.get("capital_provider"),
        "capital_recipient": row.get("capital_recipient"),
        "proposed_target_firm": audit.get("expected_target_firm"),
        "proposed_counterparty": audit.get("expected_counterparty"),
        "audit_economic_action": audit.get("economic_action"),
        "amount_text": row.get("amount_text"),
        "instrument_class": row.get("instrument_class"),
        "summary": row.get("summary"), "audit_note": audit.get("note"),
        "verbatim_quote": row.get("verbatim_quote"),
        "source_type": row.get("source_type"), "doc_date": row.get("doc_date"),
        "content_digest": review_content_digest(row),
        "decision": "", "reviewer": "", "note": "",
    }


DIRECTION_QUEUE_COLUMNS = ["raw_id", "chunk_id", "escalation_reason",
                           "target_firm", "counterparty",
                           "economic_action", "capital_provider",
                           "capital_recipient",
                           "proposed_target_firm", "proposed_counterparty",
                           "audit_economic_action",
                           "amount_text", "instrument_class",
                           "summary", "audit_note", "verbatim_quote",
                           "source_type", "doc_date", "content_digest",
                           "decision", "reviewer", "note"]


def _review_key(cache: ResultCache, row: dict, model_id: str) -> str:
    """Cache key for one row's Stage-3 audit.  The cache id is
    `raw_id|<content-digest>`, where the digest (prompts.review_content_digest)
    covers the EXACT auditor-visible fields build_review_user serializes.  Any
    assemble-layer rewrite of an auditor-visible field (the hostable screen
    nulling amount_text, a currency guard, a basis coercion, ...) therefore
    yields a fresh cache identity, so a stale verdict from a materially
    different pre-rewrite row is never replayed; byte-identical content keeps
    the cached verdict."""
    rid = f"{row.get('raw_id')}|{review_content_digest(row)}"
    return cache.make_key(rid, REVIEW_STAGE, REVIEW_VERSION, model_id)


def run_review(kept_rows: list, chunks_by_id: dict, cache: ResultCache, llm: LLM,
               model_key: str = REVIEW_MODEL, limit: Optional[int] = None,
               batch: bool = True) -> tuple:
    """Audit each kept row against its source EXCERPT. Emits a SEPARATE audit
    row per instance (keyed on raw_id): the eight per-check grades
    (ok|caveat|bad|unsure) + a note + the code-computed quote_verbatim_pass.
    NO pass/fail verdict. Resumable. Returns (audit_rows, stats)."""
    cfg = CallConfig(model_key=model_key, max_tokens=None)
    model_id = cfg.model_id
    audit_rows = []
    n_called = n_quar = 0
    grade_counts = Counter()

    rows = kept_rows[:limit] if limit else kept_rows
    # Keys THIS run already retried and re-quarantined -- final for the run
    # (no sync double-billing of the same failure); an earlier-run quarantine
    # not yet retried this run falls through and retries.
    run_quarantined = set()
    # Batch-prefill the cache misses through the Message Batches API.
    if batch and not llm.mock:
        reqs = []
        for row in rows:
            rid = row.get("raw_id")
            key = _review_key(cache, row, model_id)
            cc = cache.get(key)
            # Earlier-run quarantines retry; an OK result is final.
            if cc is not None and cc.get("status") == "ok":
                continue
            chunk = chunks_by_id.get(row.get("chunk_id"))
            chunk_text = chunk.get("text") if chunk else ""
            reqs.append(((rid, key), REVIEW_PROMPT, build_review_user(row, chunk_text)))
        if reqs:
            def _persist(rk, res):
                rid_, key_ = rk
                obj_, usage_, status_ = res
                cache.put(key_, rid_, REVIEW_STAGE, REVIEW_VERSION, model_id,
                          status_, obj_,
                          usage_["in_tokens"], usage_["out_tokens"],
                          stop_reason=usage_.get("stop_reason"),
                          failure_reason=usage_.get("failure_reason"),
                          telemetry=usage_)
            results = llm.call_json_batch(reqs, cfg, validate=valid_review,
                                          label="review", on_result=_persist)
            for (rid, key), (obj, usage, status) in results.items():
                cache.put(key, rid, REVIEW_STAGE, REVIEW_VERSION, model_id, status, obj,
                          usage["in_tokens"], usage["out_tokens"],
                          stop_reason=usage.get("stop_reason"),
                          failure_reason=usage.get("failure_reason"), telemetry=usage)
                n_called += 1
                USAGE_METER.add(REVIEW_STAGE, model_id, usage)
                if status != "ok":
                    n_quar += 1
                    run_quarantined.add(key)

    for i, row in enumerate(rows):
        rid = row.get("raw_id")
        chunk = chunks_by_id.get(row.get("chunk_id"))
        chunk_text = chunk.get("text") if chunk else ""
        key = _review_key(cache, row, model_id)
        cached = cache.get(key)
        parse_status = "ok"
        if cached is not None and (cached["status"] == "ok"
                                   or key in run_quarantined
                                   or (batch and not llm.mock)):
            if cached["status"] == "ok" and valid_review(cached.get("payload")):
                audit = cached["payload"]
            else:
                audit, parse_status = _UNPARSABLE_AUDIT, "unparsable"
        else:
            obj, usage, status = llm.call_json(
                REVIEW_PROMPT, build_review_user(row, chunk_text), cfg, validate=valid_review)
            cache.put(key, rid, REVIEW_STAGE, REVIEW_VERSION, model_id, status, obj,
                      usage["in_tokens"], usage["out_tokens"],
                      stop_reason=usage.get("stop_reason"),
                      failure_reason=usage.get("failure_reason"), telemetry=usage)
            n_called += 1
            USAGE_METER.add(REVIEW_STAGE, model_id, usage)
            if status != "ok":
                n_quar += 1
                audit, parse_status = _UNPARSABLE_AUDIT, "unparsable"
            else:
                audit = obj
            if n_called % 100 == 0:
                print(f"  review: called {n_called}, quarantined {n_quar}")
        checks = audit.get("checks") or {}
        roles = audit.get("role_resolution") or {}
        arow = {"raw_id": rid, "audit_parse_status": parse_status}
        for k in REVIEW_ROLE_KEYS:
            arow[k] = roles.get(k)
        for k in REVIEW_CHECK_KEYS:
            arow[k] = checks.get(k)
            grade_counts[(k, checks.get(k))] += 1
        arow["note"] = audit.get("note")
        arow["quote_verbatim_pass"] = _verbatim_pass(row.get("verbatim_quote"), chunk_text)
        arow["quote_verbatim_grade"], arow["quote_missing_tokens"] = \
            _verbatim_grade(row.get("verbatim_quote"), chunk_text)
        audit_rows.append(arow)
    stats = {"called": n_called, "quarantined": n_quar, "audited": len(audit_rows),
             "prompt_version": REVIEW_VERSION,
             "bad_counts": {k: grade_counts.get((k, "bad"), 0) for k in REVIEW_CHECK_KEYS},
             "caveat_counts": {k: grade_counts.get((k, "caveat"), 0) for k in REVIEW_CHECK_KEYS},
             "unsure_counts": {k: grade_counts.get((k, "unsure"), 0) for k in REVIEW_CHECK_KEYS},
             "unparsable": sum(1 for a in audit_rows if a.get("audit_parse_status") != "ok"),
             "verbatim_fail": sum(1 for a in audit_rows if not a["quote_verbatim_pass"]),
             "verbatim_grades": dict(Counter(a["quote_verbatim_grade"] for a in audit_rows))}
    return audit_rows, stats


# ===========================================================================
# The stages as separate, resumable steps (one shared cache per outdir)
# ===========================================================================
def run_stage1(chunks_path: str, outdir: str, *, screen_model: str = SCREEN_MODEL,
               mock: bool = False, preskip: bool = True,
               limit: Optional[int] = None, batch: bool = True) -> dict:
    """Stage 1 only: relevance screen -> cache."""
    ensure_dir(outdir)
    cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
    llm = LLM(mock=mock, mock_fn=_mock_fn if mock else None)
    print("== Stage 1: relevance screen ==")
    s1 = run_screen(chunks_path, cache, llm, model_key=screen_model,
                    preskip=preskip, limit=limit, batch=batch)
    print("  screen:", s1)
    cache.close()
    return s1


def seed_all_relevant(chunks_path: str, outdir: str, *, screen_model: str = SCREEN_MODEL,
                      limit: Optional[int] = None) -> int:
    """SKIP Stage 1: seed the cache with a marked 'relevant=1' verdict for
    every chunk in chunks_path that has NO real screen verdict yet, so
    run_stage2 treats a pre-screened chunk set as relevant without paying for
    the screen.  A real cached verdict (relevant or not) is never overwritten
    -- the seed only fills gaps.  Seeds carry `preseeded=True`, so a later
    REAL --screen run re-screens them instead of accepting them as screened
    output.  `limit` caps the seeded chunks for staged ramp-up runs.
    Returns the number of chunks covered (seeded + already-cached)."""
    ensure_dir(outdir)
    cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
    model_id = CallConfig(model_key=screen_model).model_id
    n = n_written = n_kept_real = 0
    for ch in read_jsonl(chunks_path):
        if limit and n >= limit:
            break
        cid = ch["chunk_id"]
        key = cache.make_key(_chunk_cache_id(ch), SCREEN_STAGE, SCREEN_VERSION,
                             model_id)
        cc = cache.get(key)
        if cc is None:
            cache.put(key, cid, SCREEN_STAGE, SCREEN_VERSION, model_id, "ok",
                      {"relevant": 1, "reason": PRESEED_REASON,
                       "preseeded": True}, 0, 0)
            n_written += 1
        elif not (cc.get("payload") or {}).get("preseeded"):
            n_kept_real += 1
        n += 1
    print(f"== Stage 1 SKIPPED: {n} chunk(s) covered -- {n_written} seeded "
          f"relevant, {n_kept_real} kept their existing REAL screen verdict, "
          f"{n - n_written - n_kept_real} already seeded ==")
    cache.close()
    return n


def run_stage2(chunks_path: str, outdir: str, *, extract_model: str = EXTRACT_MODEL,
               mock: bool = False, limit: Optional[int] = None,
               batch: bool = True) -> dict:
    """Stage 2 only (reads Stage-1 flags from the same cache): extract+classify ->
    save the ORIGINAL raw to raw_stage2.jsonl. No gating / review here."""
    ensure_dir(outdir)
    cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
    llm = LLM(mock=mock, mock_fn=_mock_fn if mock else None)
    print("== Stage 2: capital-fact extraction + classification ==")
    chunks_by_id = {ch["chunk_id"]: ch for ch in read_jsonl(chunks_path)}
    relevant = relevant_chunk_ids(cache, chunks_by_id)
    if not relevant:
        raise SystemExit(
            "no relevant chunks in cache for THIS corpus/screen-prompt/screen-"
            "model -- run Stage 1 first (run_stage1), or seed with --no-screen.")
    # One shared work list: run_extract processes it, collect_raw_instances
    # reads exactly it back -- raw_stage2.jsonl is this run's scope, nothing
    # more (no cross-corpus / cross-limit / cross-model cache leakage).
    selected = select_chunks(chunks_by_id, relevant, limit)
    print(f"  relevant chunks to extract: {len(relevant)}"
          f" (selected this run: {len(selected)})")
    s2 = run_extract(chunks_by_id, set(selected), cache, llm,
                     model_key=extract_model, limit=None, batch=batch)
    print("  extract:", s2)
    extract_model_id = CallConfig(model_key=extract_model).model_id
    raw = collect_raw_instances(cache, chunks_by_id, selected, extract_model_id)
    write_jsonl(raw, os.path.join(outdir, "raw_stage2.jsonl"))
    n_salvaged = sum(1 for o in raw if o.get("_truncated_salvage"))
    if n_salvaged:
        print(f"  WARNING: {n_salvaged} raw observation(s) came from truncated-output "
              f"salvage -- inspect run_stats/truncated_chunks")
    print(f"  raw_stage2 observations: {len(raw)}")
    cache.close()
    return {"extract": s2, "n_raw": len(raw), "salvaged_rows_in_raw": n_salvaged}


def run_stage3(chunks_path: str, outdir: str, *, review_model: str = REVIEW_MODEL,
               mock: bool = False, include_news: bool = True,
               limit: Optional[int] = None, batch: bool = True) -> dict:
    """Stage 3: classify + GATE the raw (dropped.jsonl) -> AUDIT the kept rows
    (verify Stage-2 fields, set review_pass / flags) -> write the ONE table."""
    ensure_dir(outdir)
    cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
    llm = LLM(mock=mock, mock_fn=_mock_fn if mock else None)
    t0 = time.time()

    raw_path = os.path.join(outdir, "raw_stage2.jsonl")
    if not os.path.exists(raw_path):
        raise SystemExit("no raw_stage2.jsonl -- run Stage 2 first (run_stage2).")
    raw = list(read_jsonl(raw_path))

    print("== Stage 3a: classify + gate ==")
    g = D.classify_and_gate(raw, include_news=include_news)
    kept, dropped = g["kept"], g["dropped"]
    drop_counts = Counter(r.get("drop_reason") for r in dropped)
    print(f"  raw {len(raw)} -> kept {len(kept)} | dropped {len(dropped)} {dict(drop_counts)}")
    if g.get("n_defective"):
        rate = g["n_defective"] / len(raw) if raw else 0.0
        print(f"  contract defects (recorded, not repaired): "
              f"{g['n_defective']}/{len(raw)} observations ({rate:.1%}) | "
              f"by class: {g['defect_counts']}")

    print("== Stage 3b: defensive review (audit only) ==")
    chunks_by_id = {ch["chunk_id"]: ch for ch in read_jsonl(chunks_path)}
    audit_rows, s3 = run_review(kept, chunks_by_id, cache, llm, model_key=review_model,
                                limit=limit, batch=batch)
    print("  review:", s3)

    # Admission contract:
    #   * an unparsable audit, or ANY bad check other than time_ok /
    #     anchor_direction, drops the row (stage3_audit_bad);
    #   * anchor_direction as the ONLY fatal-class bad routes the row to the
    #     HUMAN direction-review lane: with a matching human_overrides.jsonl
    #     record the corrected row is admitted (direction_fix_applied=True),
    #     otherwise it is EXCLUDED (fail-closed) and queued with Stage-3's own
    #     expected_* roles pre-filled as the proposed fix;
    #   * time_ok bad alone does not drop (the time layer owns time): the row
    #     is admitted with audit_time_bad=True for the time layer to re-derive.
    # The auditor's own EVIDENCE also routes deterministically, its grades
    # untouched:
    #   * direction_escalation (expected roles contradict the row, or a named
    #     counterparty was omitted) joins the SAME fail-closed direction lane
    #     even when the grade is only caveat/ok;
    #   * an explicit should=<different value> in the note admits the row with
    #     audit_should_mismatch set (review reason, never a drop).
    # Stage 3 itself never edits fields -- only a signed human override does.
    overrides = load_overrides(outdir)
    audit_by_id = {a["raw_id"]: a for a in audit_rows}
    admitted, direction_pending, auto_fixes = [], [], []
    n_override_applied = n_escalated = n_auto_fixed = 0
    for row in kept:
        rid = str(row.get("raw_id") or "")
        audit = audit_by_id.get(row.get("raw_id")) or {}
        parse_status = audit.get("audit_parse_status")
        bad_fields = [key for key in REVIEW_CHECK_KEYS if audit.get(key) == "bad"]
        fatal_fields = [k for k in bad_fields
                        if k not in NONFATAL_CHECKS and k != "anchor_direction"]
        if parse_status != "ok" or fatal_fields:
            rejected = dict(row)
            rejected["drop_reason"] = ("auditor_unparsable" if parse_status != "ok"
                                       else "stage3_audit_bad")
            rejected["audit_bad_fields"] = fatal_fields or bad_fields
            rejected["audit_note"] = audit.get("note")
            dropped.append(rejected)
            continue
        escalation = (direction_escalation(audit, row) or emitted_role_mismatch(row)
                      or inverted_anchor(row) or degenerate_slots(row))
        # A spurious counterparty produces neither an anchor_direction bad nor an
        # escalation (the row names a counterparty, and one emitted role is null,
        # so emitted_role_mismatch skips it) -- admit it into this branch on its
        # own signal, or the consensus-clear path below is unreachable.
        clear = consensus_clear_counterparty(audit, row)
        if "anchor_direction" in bad_fields or escalation or clear:
            override = overrides.get(rid)
            # Content binding: an override signed against a materially
            # different version of this row (re-extraction under a new prompt
            # can reproduce the same raw_id with different slots) must not
            # replay -- the row goes back to the queue instead.
            if override is not None and override.get("content_digest") \
                    and override["content_digest"] != review_content_digest(row):
                print(f"  WARNING: stale direction override for {rid} "
                      f"(row content changed since review) -- override ignored")
                override = None
            if override is not None:
                fixed = apply_direction_override(row, override)
                # Post-override coherence: a set_roles decision can itself
                # create target == counterparty; never admit that unchecked.
                if degenerate_slots(fixed):
                    print(f"  WARNING: direction override for {rid} produces "
                          f"target == counterparty -- re-queued")
                    item = _direction_queue_row(row, audit)
                    item["escalation_reason"] = "override_created_degenerate_slots"
                    direction_pending.append(item)
                    continue
                row = fixed
                row["audit_time_bad"] = "time_ok" in bad_fields
                row["audit_should_mismatch"] = should_mismatch(audit, row)
                admitted.append(row)
                n_override_applied += 1

            elif auto_direction_fix(audit, row):
                before_t, before_c = row.get("target_firm"), row.get("counterparty")
                row = dict(row)
                row["target_firm"], row["counterparty"] = before_c, before_t
                row["direction_fix_applied"] = True
                row["direction_fix_source"] = "model_consensus"
                row["audit_time_bad"] = "time_ok" in bad_fields
                row["audit_should_mismatch"] = should_mismatch(audit, row)
                auto_fixes.append({
                    "raw_id": rid, "chunk_id": row.get("chunk_id"),
                    "direction_fix_source": "model_consensus",
                    "escalation_reason": ("anchor_direction_bad"
                                          if "anchor_direction" in bad_fields else escalation),
                    "before_target_firm": before_t, "before_counterparty": before_c,
                    "after_target_firm": row.get("target_firm"),
                    "after_counterparty": row.get("counterparty"),
                    "economic_action": row.get("economic_action") or audit.get("economic_action"),
                    "capital_provider": row.get("capital_provider"),
                    "capital_recipient": row.get("capital_recipient"),
                    "audit_expected_target": audit.get("expected_target_firm"),
                    "audit_expected_counterparty": audit.get("expected_counterparty"),
                    "amount_text": row.get("amount_text"),
                    "instrument_class": row.get("instrument_class"),
                    "source_type": row.get("source_type"), "doc_date": row.get("doc_date"),
                    "verbatim_quote": row.get("verbatim_quote"),
                    "audit_note": audit.get("note"),
                })
                admitted.append(row)
                n_auto_fixed += 1

            elif consensus_clear_counterparty(audit, row):
                # Both models agree the named counterparty is not an endpoint of
                # this flow (bookrunners / arrangers / advisers).  Move them to
                # other_parties -- never delete: they are real participants.
                before_c = row.get("counterparty")
                row = dict(row)
                moved = [p.strip() for p in str(before_c or "").split(";") if p.strip()]
                existing = row.get("other_parties")
                if isinstance(existing, str):
                    try:
                        existing = json.loads(existing)
                    except Exception:
                        existing = []
                seen, merged_parties = set(), []
                for p in list(existing or []) + moved:
                    if p and str(p).lower() not in seen:
                        seen.add(str(p).lower())
                        merged_parties.append(p)
                row["other_parties"] = merged_parties
                row["counterparty"] = None
                row["direction_fix_applied"] = True
                row["direction_fix_source"] = "model_consensus_clear"
                row["audit_time_bad"] = "time_ok" in bad_fields
                row["audit_should_mismatch"] = should_mismatch(audit, row)
                auto_fixes.append({
                    "raw_id": rid, "chunk_id": row.get("chunk_id"),
                    "direction_fix_source": "model_consensus_clear",
                    "escalation_reason": escalation or "anchor_direction_bad",
                    "before_target_firm": row.get("target_firm"),
                    "before_counterparty": before_c,
                    "after_target_firm": row.get("target_firm"),
                    "after_counterparty": None,
                    "moved_to_other_parties": moved,
                    "economic_action": row.get("economic_action") or audit.get("economic_action"),
                    "capital_provider": row.get("capital_provider"),
                    "capital_recipient": row.get("capital_recipient"),
                    "audit_expected_counterparty": audit.get("expected_counterparty"),
                    "amount_text": row.get("amount_text"),
                    "instrument_class": row.get("instrument_class"),
                    "source_type": row.get("source_type"), "doc_date": row.get("doc_date"),
                    "verbatim_quote": row.get("verbatim_quote"),
                    "audit_note": audit.get("note"),
                })
                admitted.append(row)
                n_auto_fixed += 1

            elif degenerate_slot_fix(audit, row):
                # target == counterparty is self-contradiction; DIR-4 cannot
                # fire on it (its guard needs tgt != cp).  Extractor roles and
                # auditor pair independently agree on the same correction, so
                # only the duplicated counterparty slot is rewritten -- with
                # the extractor's own surface -- and the target never moves.
                fixed_cp = degenerate_slot_fix(audit, row)
                before_c = row.get("counterparty")
                row = dict(row)
                row["counterparty"] = fixed_cp
                row["direction_fix_applied"] = True
                row["direction_fix_source"] = "model_consensus_degenerate"
                row["audit_time_bad"] = "time_ok" in bad_fields
                row["audit_should_mismatch"] = should_mismatch(audit, row)
                auto_fixes.append({
                    "raw_id": rid, "chunk_id": row.get("chunk_id"),
                    "direction_fix_source": "model_consensus_degenerate",
                    "escalation_reason": escalation or "anchor_direction_bad",
                    "before_target_firm": row.get("target_firm"),
                    "before_counterparty": before_c,
                    "after_target_firm": row.get("target_firm"),
                    "after_counterparty": fixed_cp,
                    "economic_action": row.get("economic_action") or audit.get("economic_action"),
                    "capital_provider": row.get("capital_provider"),
                    "capital_recipient": row.get("capital_recipient"),
                    "audit_expected_counterparty": audit.get("expected_counterparty"),
                    "amount_text": row.get("amount_text"),
                    "instrument_class": row.get("instrument_class"),
                    "source_type": row.get("source_type"), "doc_date": row.get("doc_date"),
                    "verbatim_quote": row.get("verbatim_quote"),
                    "audit_note": audit.get("note"),
                })
                admitted.append(row)
                n_auto_fixed += 1

            else:
                if "anchor_direction" not in bad_fields:
                    n_escalated += 1
                item = _direction_queue_row(row, audit)
                item["escalation_reason"] = ("anchor_direction_bad"
                                             if "anchor_direction" in bad_fields
                                             else escalation)
                direction_pending.append(item)
            continue
        
        row = dict(row)
        row["audit_time_bad"] = "time_ok" in bad_fields
        row["audit_should_mismatch"] = should_mismatch(audit, row)
        admitted.append(row)
    n_rejected = len(kept) - len(admitted) - len(direction_pending)
    kept = admitted
    write_jsonl(direction_pending, os.path.join(outdir, DIRECTION_QUEUE_NAME + ".jsonl"))
    write_csv(direction_pending, DIRECTION_QUEUE_COLUMNS,
              os.path.join(outdir, DIRECTION_QUEUE_NAME + ".csv"))
    write_jsonl(auto_fixes, os.path.join(outdir, "auto_direction_fixes.jsonl"))
    n_time_bad = sum(1 for r in kept if r.get("audit_time_bad"))
    n_should = sum(1 for r in kept if r.get("audit_should_mismatch"))
    print(f"  audit admission: dropped {n_rejected} (fatal bad/unparsable), "
          f"direction-review pending {len(direction_pending)} "
          f"(of which {n_escalated} deterministically escalated; excluded until human override), "
          f"overrides applied {n_override_applied}, auto direction-fixed {n_auto_fixed}, "
          f"time-bad admitted+flagged {n_time_bad}, "
          f"should-mismatch flagged {n_should} "
          f"-> kept {len(kept)}")

    drop_counts = Counter(r.get("drop_reason") for r in dropped)
    write_jsonl(dropped, os.path.join(outdir, "dropped.jsonl"))

    # Sampled false-drop re-check lane (additive output for a human).
    spotcheck = D.sample_drop_spotcheck(dropped, DROP_SPOTCHECK_RATE)
    write_jsonl(spotcheck, os.path.join(outdir, "drop_spotcheck.jsonl"))
    if spotcheck:
        print(f"  drop spot-check: sampled {len(spotcheck)} strong-signal drop(s) "
              f"@ rate {DROP_SPOTCHECK_RATE} -> drop_spotcheck.jsonl")

    D.sort_rows(kept)
    write_csv(kept, D.INSTANCE_COLUMNS, os.path.join(outdir, "capital_instances.csv"))
    write_jsonl(kept, os.path.join(outdir, "capital_instances.jsonl"))

    # The audit is a SEPARATE table, linked to instances by raw_id (no verdict).
    audit_columns = ["raw_id", "audit_parse_status", *REVIEW_ROLE_KEYS,
                     *REVIEW_CHECK_KEYS, "quote_verbatim_pass",
                     "quote_verbatim_grade", "note"]
    write_csv(audit_rows, audit_columns, os.path.join(outdir, "audit.csv"))
    write_jsonl(audit_rows, os.path.join(outdir, "audit.jsonl"))

    class_counts = Counter(r.get("instrument_class") for r in kept)
    print(f"  capital_instances rows: {len(kept)} | audit rows: {len(audit_rows)}")
    print(f"  instrument_class : {dict(class_counts)}")
    print(f"  per-check bad counts : {s3['bad_counts']}")
    print(f"  verbatim_fail : {s3['verbatim_fail']}")

    drop_counts = Counter(r.get("drop_reason") for r in dropped)
    # t0 starts inside run_stage3, so stage3_elapsed_s times ONLY Stage 3
    # (fractions of a second on a full cache hit).
    stats = {"stage3_elapsed_s": round(time.time() - t0, 1), "review": s3,
             "n_raw": len(raw), "n_kept": len(kept), "n_dropped": len(dropped),
             "drop_counts": dict(drop_counts),
             # 2026-08-28 policy: structural contract defects are admitted,
             # defaulted/cleared, and REPORTED -- these are the paper's
             # extraction error rates (per-row detail in schema_defects).
             "schema_defect_counts": g.get("defect_counts", {}),
             "observations_with_defects": g.get("n_defective", 0),
             "schema_defect_rate": round(
                 g.get("n_defective", 0) / len(raw), 4) if raw else 0.0,
             # sampled strong-signal drops for the human false-drop re-check
             "drop_spotcheck_sampled": len(spotcheck),
             "drop_spotcheck_rate": DROP_SPOTCHECK_RATE,
             "instrument_class_counts": dict(class_counts),
             # admission telemetry
             "direction_review_pending": len(direction_pending),
             "direction_overrides_applied": n_override_applied,
             "time_bad_admitted": n_time_bad,
             "truncation_salvaged_rows_in_raw": sum(
                 1 for o in raw if o.get("_truncated_salvage")),
             "include_news": include_news, "cache_stats": cache.stats(),
             "cache_failure_stats": cache.failure_stats(),
             # per-stage model provenance: the run's instrument-version record
             "stage_models": stage_models(),
             # E1/VAL-3: live meter over screen/extract/review fresh calls
             # (list price; Batch traffic bills at 50%)
             "llm_usage": USAGE_METER.snapshot(mock=llm.mock)}
    with open(os.path.join(outdir, "run_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print("DONE:", json.dumps({k: stats[k] for k in ("stage3_elapsed_s", "n_kept")}))
    cache.close()
    return stats


# ===========================================================================
# Mock model -- exercises all stages with no key. Stage detected from a stable
# phrase in each system prompt.
# ===========================================================================
_NEXUS_RE = re.compile(
    r"data[\s-]*cent|server|gpu|\bcompute\b|cloud|\bOCI\b|\bAzure\b|\bAWS\b|\bAI\b"
    r"|gigawatt|megawatt|\bMW\b|\bGW\b|power|chips?|nvidia|supercomput", re.I)


def _mock_instrument_class(low: str) -> str:
    """Mock instrument_class by precedence. Wiring only -- not meant to be accurate."""
    if any(k in low for k in ("variable interest", "vie", "spv", "joint venture",
                              "membership interest", "the venture", "securitization",
                              "maximum exposure")):
        return "jv_spv"
    if any(k in low for k in ("additions to property", "purchases of property",
                              "construction in progress", "capital expenditure", "capex")):
        return "general_capex"
    if any(k in low for k in ("construction loan", "project financ")):
        return "project_debt"
    if any(k in low for k in ("senior notes", "long-term debt", "term loan", "notes due",
                              "aggregate principal")):
        return "corporate_debt"
    if any(k in low for k in ("guarantee", "letter of credit", "backstop", "indemnif")):
        return "guarantee"
    if any(k in low for k in ("power purchase", "ppa", "gigawatt", "megawatt",
                              " gw", " mw", "offtake")):
        return "ppa_power"
    if "lease" in low:
        return "lease"
    if "purchase" in low:
        return "supplier_purchase"
    return "other"


def _mock_fn(system: str, user: str, cfg) -> str:
    # ---- Stage 3: review (8 checks ok|caveat|bad|unsure, NO verdict) ----
    if "stage-3 AUDITOR" in system:
        try:
            candidate = json.loads(
                user.split("EXTRACTED INSTANCE (audit this):", 1)[1].strip())
        except Exception:
            candidate = {}
        return json.dumps({
            "role_resolution": {
                "economic_action": candidate.get("economic_action") or "unclear",
                "expected_capital_provider": candidate.get("capital_provider"),
                "expected_capital_recipient": candidate.get("capital_recipient"),
                "expected_target_firm": candidate.get("target_firm"),
                "expected_counterparty": candidate.get("counterparty")},
            "checks": {k: "ok" for k in REVIEW_CHECK_KEYS},
            "note": "mock"})

    text = user.split("EXCERPT:", 1)[-1]
    low = text.lower()
    has_num = bool(re.search(r"[\d,]+(?:\.\d+)?\s*(?:billion|million|bn|gigawatt|megawatt|gw|mw)", low))

    # ---- Stage 1: screen ----  (Stage 2's prompt names itself "stage-2 extractor")
    if "stage-2 extractor" not in system:
        kw = any(k in low for k in
                 ["lease", "capital expenditure", "capex", "guarantee", "variable interest",
                  "senior notes", "power purchase", "power", "construction in progress",
                  "commenced", "purchase", "agreement", "data center", "gigawatt", "megawatt", "azure"])
        return json.dumps({"relevant": 1 if (kw and has_num) else 0, "reason": "mock"})

    # ---- Stage 2: extract + classify ----
    _money = re.findall(r"\$\s?[\d,]+(?:\.\d+)?\s*(?:billion|million|bn)?"
                        r"|\b[\d,]+(?:\.\d+)?\s+(?:billion|million|bn)\b", text, re.I)
    amt = _money[0].strip() if _money else None
    pw = re.search(r"([\d.]+)\s*(gigawatts?|megawatts?|gw|mw)\b", low)
    if amt is None and not pw:
        return json.dumps({"observations": []})

    klass = _mock_instrument_class(low)
    # Keep the mock quote centered on the selected amount so strict evidence
    # validation exercises the same grounding contract as a real response.
    anchor = text.find(amt) if amt else (pw.start() if pw else 0)
    quote = text[max(0, anchor - 180): anchor + 360].strip()
    obj_name = next((p for p in ["Hyperion", "Lake Mariner", "Stargate", "Fairwater"]
                     if p in quote), None)
    scope = "entire_amount" if _NEXUS_RE.search(quote) else "unclear"

    mh = re.search(r"FILER:\s*([^|]+)", user)
    filer = V.canonical_firm(mh.group(1).strip()) if mh else None
    target = None
    for tok in re.findall(r"amazon web services|google cloud|[A-Za-z]+", quote):
        c = V.canonical_firm(tok)
        if c:
            target = c; break
    target = target or filer or "Mock Firm"
    cp = "Blue Owl" if obj_name else None
    action = ("financing" if klass in {"jv_spv", "equity_contribution",
                                        "project_debt", "corporate_debt",
                                        "mixed_financing", "grant_subsidy"}
              else "guarantee" if klass == "guarantee"
              else "acquisition" if klass in {"m_and_a", "asset_acquisition"}
              else "spending")
    provider = cp if action == "financing" else target
    recipient = target if action == "financing" else cp
    evidence = {k: None for k in (
        "target", "amount_or_capacity", "amount_basis", "amount_bound_type",
        "amount_aggregation_level", "instrument", "debt_subtype", "counterparty",
        "object", "parent_object", "claim_source", "ai_nexus",
        "period_or_horizon", "deal_status", "commitment_status", "parent")}

    base = {"target_firm": target, "target_parent": None,
            "target_parent_relation": "na",
             "currency": "USD",
            "amount_bound_type": "exact", "amount_aggregation_level": "object_total",
             "quantity_value": None, "quantity_unit": None,
             "temporal_context": None,
            "object_name": obj_name,
            "object_type": ("data_center_project" if obj_name else None),
            "object_evidence": (obj_name if obj_name else None),
            "parent_object_name": None, "parent_object_type": None,
            "parent_object_relation": None, "parent_object_evidence": None,
            "counterparty": cp, "economic_action": action,
            "capital_provider": provider, "capital_recipient": recipient,
            "other_parties": [],
             "ai_infra_scope": scope, "geo_scope": "unknown",
             "instrument_class": klass,
            "debt_instrument_subtype": ("unknown" if klass in
                                         {"corporate_debt", "project_debt"} else None),
            "claim_source": "legal_or_filed_record",
            "claim_source_name": "mock source",
            "deal_status": "new_commitment", "deal_status_cue": None,
            "commitment_status": "unclear", "commitment_status_cue": None,
            "conditionality": "unclear", "condition_text": None,
            "method_summary": "mock instrument",
            "verbatim_quote": quote, "field_evidence": evidence,
            "summary": "mock instance"}

    if pw and (amt is None or "$" not in amt):
        unit = {"gigawatt": "GW", "gigawatts": "GW", "gw": "GW",
                "megawatt": "MW", "megawatts": "MW", "mw": "MW"}[pw.group(2)]
        base.update({"amount_text": None, "amount_scale": "none",
                     "amount_basis": "capacity_only",
                     "quantity_value": float(pw.group(1)), "quantity_unit": unit,
                     "method_summary": "mock power capacity"})
        return json.dumps({"observations": [base]})

    _basis = ("debt_principal" if klass in ("corporate_debt", "project_debt")
              else "equity_capital" if klass == "jv_spv"
              else "capex_cash_flow" if klass == "general_capex"
              else "recognized_balance" if klass == "lease"
              else "undiscounted_commitment")
    base.update({"amount_text": amt, "amount_basis": _basis,
                 "amount_scale": ("billion" if amt and re.search(r"billion|bn", amt, re.I)
                                  else "million" if amt and re.search(r"million", amt, re.I)
                                  else "full_units")})
    return json.dumps({"observations": [base]})
