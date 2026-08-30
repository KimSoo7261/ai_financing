"""Run-time progress reporting: stage timers, timed heartbeats, run-end table.

Console policy (2026-08-20 user decision): AGGREGATE lines plus a heartbeat at
most every 30 seconds inside the long sequential loops -- per-decision detail
is never printed; it already persists in the link ledgers
(entity_links.jsonl / object_links.jsonl / fact_links.jsonl).

Heartbeats are time-based (default 30s; override with AICAP_HEARTBEAT_S, e.g.
=0 to print on every eligible beat when testing).  A fast fixture run prints
no heartbeat at all -- silence means "faster than the interval", not "hung".
"""
from __future__ import annotations

import os
import time


def fmt_secs(s: float | int | None) -> str:
    """12.3s / 4m12s / 1h04m -- compact wall-clock for humans."""
    if s is None:
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class Heartbeat:
    """Prints `label: done/total in Xs | ~Ys left | <extra>` at most every
    `interval_s` seconds.  `extra_fn` is only evaluated when a line actually
    prints, so per-item cost is one time() call and one comparison."""

    def __init__(self, label: str, total: int, interval_s: float = 30.0):
        self.label = label
        self.total = int(total)
        self.interval = float(os.environ.get("AICAP_HEARTBEAT_S", interval_s))
        self.t0 = time.monotonic()
        self._last = self.t0            # first line after one full interval

    def beat(self, done: int, extra_fn=None) -> None:
        now = time.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now
        elapsed = now - self.t0
        msg = f"   {self.label}: {done}/{self.total} in {fmt_secs(elapsed)}"
        if done > 0 and elapsed > 0:
            rate = done / elapsed
            msg += f" | ~{fmt_secs((self.total - done) / rate)} left"
        if extra_fn is not None:
            try:
                msg += f" | {extra_fn()}"
            except Exception:           # progress must never kill the stage
                pass
        print(msg, flush=True)


def print_run_table(stages: dict, llm: dict) -> None:
    """Run-end summary: one aligned row per stage (elapsed, registry-judge
    call/cache/cost deltas), then the LLM grand total."""
    rows = []
    for name, st in stages.items():
        if not isinstance(st, dict):
            continue
        jd = st.get("judge_delta") or {}
        rows.append((name, st.get("elapsed_s"), jd.get("calls"),
                     jd.get("cache_hits"), jd.get("est_cost")))
    if not rows:
        return
    print("\n== run summary " + "=" * 55)
    print(f"{'stage':30}{'elapsed':>9}{'judge':>7}{'cached':>8}{'~cost':>9}")
    total_elapsed = 0.0
    for name, el, calls, hits, cost in rows:
        total_elapsed += float(el or 0)
        print(f"{name:30}{fmt_secs(el):>9}"
              f"{('-' if calls is None else calls):>7}"
              f"{('-' if hits is None else hits):>8}"
              f"{('-' if cost is None else f'{cost:.2f}'):>9}")
    tot_calls = llm.get("_total_calls_including_time", llm.get("_total_calls", "-"))
    tot_hits = llm.get("_total_cache_hits_including_time",
                       llm.get("_total_cache_hits", "-"))
    cost = llm.get("_est_cost")
    print(f"{'TOTAL':30}{fmt_secs(total_elapsed):>9}{tot_calls:>7}{tot_hits:>8}"
          f"{('-' if not isinstance(cost, (int, float)) else f'{cost:.2f}'):>9}"
          f" {llm.get('_currency', '')}")
    print("per-decision ledgers in the run dir: entity_links.jsonl / "
          "object_links.jsonl / fact_links.jsonl", flush=True)
