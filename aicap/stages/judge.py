#!/usr/bin/env python3
"""The Judge: the cached LLM adjudicator used by the registry stages.

Wraps the shared aicap.llm transport (retries, JSON repair, Message Batches
API) behind a Judge that:
  * checks the resumable sqlite cache first (key: item content hash + stage +
    prompt version + model id -- content-stable, order-free);
  * counts calls / hits / tokens / estimated cost PER STAGE;
  * returns None on quarantine (caller treats as abstain -> review, never a
    forced decision);
  * batches independent judgments through call_json_batch when >= BATCH_MIN
    items are uncached.

The time-extraction stage is the one registry stage that does NOT go through
the Judge: it drives aicap.llm directly via aicap.time_normalizer (same cache
file, its own stage keys).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from ..config import MODELS
from ..llm import LLM, CallConfig, ResultCache, prompt_hash

from .common import REG_MODEL, sha16

BATCH_MIN = 4          # below this, synchronous calls (no batch poll latency)
POLL_SECONDS = 15


class Judge:
    def __init__(self, outdir: str, mock: bool = False,
                 mock_fn: Optional[Callable] = None, model_key: str = REG_MODEL,
                 batch: bool = True):
        self.cache = ResultCache(os.path.join(outdir, "cache.sqlite"))
        self.llm = LLM(mock=mock, mock_fn=mock_fn)
        self.model_key = model_key
        self.cfg = CallConfig(model_key=model_key)
        self.model_id = self.cfg.model_id
        # batch=False forces every judgment through the synchronous path:
        # half the discount, but bounded, watchable wall-time instead of
        # Message-Batches queue latency (--no-batch now reaches the registry).
        self.batch = batch
        self.stats: dict[str, dict] = {}

    def close(self) -> None:
        self.cache.close()

    # ---- accounting ------------------------------------------------------
    def _st(self, stage: str) -> dict:
        return self.stats.setdefault(stage, {
            "calls": 0, "cache_hits": 0, "quarantines": 0,
            "in_tokens": 0, "out_tokens": 0})

    def _record(self, stage: str, usage: dict, status: str) -> None:
        st = self._st(stage)
        st["calls"] += 1
        st["in_tokens"] += int(usage.get("in_tokens") or 0)
        st["out_tokens"] += int(usage.get("out_tokens") or 0)
        if status != "ok":
            st["quarantines"] += 1

    def projected_cost(self) -> float:
        """Cost at the model's LIST price, in its own currency.  Batch Chat
        traffic actually bills at 50% of this -- treat it as an upper bound."""
        spec = MODELS[self.model_key]
        tin = sum(s["in_tokens"] for s in self.stats.values())
        tout = sum(s["out_tokens"] for s in self.stats.values())
        return round((tin * spec["in"] + tout * spec["out"]) / 1e6, 4)

    def cost(self) -> float:
        return 0.0 if self.llm.mock else self.projected_cost()

    def report(self) -> dict:
        out = {k: dict(v) for k, v in self.stats.items()}
        out["_total_calls"] = sum(s["calls"] for s in self.stats.values())
        out["_total_cache_hits"] = sum(s["cache_hits"] for s in self.stats.values())
        out["_est_cost"] = self.cost()
        out["_currency"] = MODELS[self.model_key].get("currency", "USD")
        if self.llm.mock:
            out["_mock_projected_cost"] = self.projected_cost()
        out["_model_id"] = self.model_id
        return out

    # ---- single cached judgment -----------------------------------------
    def cached_json(self, stage: str, item_id: str, system: str, user: str,
                    validate: Optional[Callable[[Any], bool]] = None) -> Optional[Any]:
        """One bounded judgment. item_id must already be content-stable
        (caller includes a content hash). Returns parsed obj or None."""
        version = f"{stage}_{prompt_hash(system)}"
        key = ResultCache.make_key(item_id, stage, version, self.model_id)
        hit = self.cache.get(key)
        if hit is not None:
            self._st(stage)["cache_hits"] += 1
            return hit["payload"] if hit["status"] == "ok" else None
        obj, usage, status = self.llm.call_json(system, user, self.cfg,
                                                validate=validate)
        self._record(stage, usage, status)
        self.cache.put(key, item_id, stage, version, self.model_id, status,
                       obj, usage.get("in_tokens", 0), usage.get("out_tokens", 0),
                       stop_reason=usage.get("stop_reason"),
                       failure_reason=usage.get("failure_reason"), telemetry=usage)
        return obj if status == "ok" else None

    # ---- batched independent judgments ----------------------------------
    def batch_json(self, stage: str, requests: list, system: str,
                   validate: Optional[Callable[[Any], bool]] = None,
                   validators: Optional[dict] = None,
                   label: str = "") -> dict[str, Optional[Any]]:
        """requests = [(item_id, user_text), ...] sharing one system prompt.
        Returns {item_id: obj|None}. Cache-first; uncached items go through
        the Message Batches API when >= BATCH_MIN, else synchronously."""
        version = f"{stage}_{prompt_hash(system)}"
        out: dict[str, Optional[Any]] = {}
        todo = []
        for item_id, user in requests:
            key = ResultCache.make_key(item_id, stage, version, self.model_id)
            hit = self.cache.get(key)
            if hit is not None:
                self._st(stage)["cache_hits"] += 1
                out[item_id] = hit["payload"] if hit["status"] == "ok" else None
            else:
                todo.append((item_id, system, user))

        def _v(rid):
            return (validators or {}).get(rid, validate)

        if todo:
            if self.batch and len(todo) >= BATCH_MIN:
                def _persist(item_id, res):
                    # Write-through per completed item (interrupt-safe); the
                    # bookkeeping loop below rewrites the same rows.
                    obj, usage, status = res
                    key = ResultCache.make_key(item_id, stage, version,
                                               self.model_id)
                    self.cache.put(key, item_id, stage, version, self.model_id,
                                   status, obj, usage.get("in_tokens", 0),
                                   usage.get("out_tokens", 0),
                                   stop_reason=usage.get("stop_reason"),
                                   failure_reason=usage.get("failure_reason"),
                                   telemetry=usage)
                results = self.llm.call_json_batch(
                    todo, self.cfg, validate=validate,
                    poll_seconds=POLL_SECONDS, label=label or stage,
                    validators=validators, on_result=_persist)
            else:
                results = {rid: self.llm.call_json(sysm, user, self.cfg,
                                                   validate=_v(rid))
                           for rid, sysm, user in todo}
            for item_id, _sys, _user in todo:
                obj, usage, status = results[item_id]
                self._record(stage, usage, status)
                key = ResultCache.make_key(item_id, stage, version, self.model_id)
                self.cache.put(key, item_id, stage, version, self.model_id,
                               status, obj,
                               usage.get("in_tokens", 0), usage.get("out_tokens", 0),
                               stop_reason=usage.get("stop_reason"),
                               failure_reason=usage.get("failure_reason"),
                               telemetry=usage)
                out[item_id] = obj if status == "ok" else None
        return out


def content_id(prefix: str, *parts: Any) -> str:
    """Content-stable cache item id (order-free inputs hashed)."""
    return f"{prefix}|{sha16(*parts)}"
