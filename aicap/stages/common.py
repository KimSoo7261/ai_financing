#!/usr/bin/env python3
"""Shared plumbing for the Fact-registry stages.

This module is production infrastructure for the whole registry half of the
pipeline: every stage depends on it.  Everything here is deterministic --
atomic IO, name normalization, retrieval primitives (tokens / trigrams /
IDF), amount buckets, quarter helpers, and the per-stage manifest.  No LLM
calls, no state mutation outside the run directory.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from .. import time_rules as TR
# REG_MODEL (the registry adjudicator's model key) is assigned by the model
# profile in aicap.config; re-exported here for the stage modules.
from ..config import MODELS, REG_MODEL, SCHEMA_VERSION, SOURCE_RANK
from ..paths import ROOT as PIPELINE_ROOT, RUN_TAG

# ---- measure families shared by Fact state and Fact views.  Amounts are
# only ever aggregated WITHIN one family (and one basis / currency / scope);
# families never sum with each other.
FAMILY_OF_BASIS = {
    "debt_principal": "financing_flow", "equity_capital": "financing_flow",
    "financing_total": "financing_flow", "grant_subsidy": "financing_flow",
    "facility_commitment": "financing_capacity",
    "capex_cash_flow": "capital_flow",
    "acquisition_consideration": "capital_flow",
    "contract_value": "contract_value",
    "deposit_or_prepayment": "capital_flow",
    "undiscounted_commitment": "commitment_stock",
    "carrying_value": "balance_stock", "recognized_balance": "balance_stock",
    "valuation": "valuation",
    "headline_project_cost": "project_attribute",
    "investment_program_amount": "program_attribute",
    "authorized_amount": "project_attribute", "max_exposure": "project_attribute",
    "capacity_only": "capacity",
    "other": "unclassified", "unknown": "unclassified",
}
POWER_TO_MW = {"mw": 1.0, "gw": 1000.0, "kw": 0.001, "tw": 1_000_000.0}


def family_of(basis) -> str:
    return FAMILY_OF_BASIS.get((basis or "unknown").lower(), "unclassified")


# ---------------------------------------------------------------------------
# IO (atomic writes: tmp in the same directory, then os.replace)
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _atomic(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_jsonl(rows: Iterable[dict], path: str) -> int:
    rows = list(rows)
    _atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def write_json(value: Any, path: str) -> None:
    _atomic(path, json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def sha16(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def file_sha(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Nullish / numeric / date / id-list helpers
# ---------------------------------------------------------------------------
def is_nullish(v: Any) -> bool:
    return v is None or (isinstance(v, str)
                         and v.strip().lower() in {"", "null", "none", "na", "nan"})


def coerce_num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if not v:
            return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def parse_date(v: Any) -> Optional[dt.date]:
    if is_nullish(v):
        return None
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def split_semi(v: Any) -> list[str]:
    """';'-joined multi-value field -> ordered surface list (junk dropped)."""
    if is_nullish(v):
        return []
    out = []
    for part in str(v).split(";"):
        s = part.strip()
        if s and s.lower() not in ("null", "none", "na"):
            out.append(s)
    return out


def parse_name_list(v: Any) -> list[str]:
    """target_parent / other_parties: string OR JSON-list-string OR list.

    ENT-3: every element is additionally split on ';'.  target/counterparty
    were always split upstream (split_semi), but parents and other_parties
    were not, so a 'Microsoft;OpenAI' parent registered as ONE entity mention
    -- the root of the composite-string registry contamination (RE-E0010)."""
    if is_nullish(v):
        return []
    if isinstance(v, (list, tuple)):
        items = v
    else:
        s = str(v).strip()
        if s.startswith("["):
            try:
                items = json.loads(s)
            except (ValueError, TypeError):
                items = [s]
        else:
            items = [s]
    out = []
    for n in items:
        out.extend(split_semi(n))
    return out


# ---------------------------------------------------------------------------
# Name normalization + retrieval primitives (entity/object registries)
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "llc", "lp", "llp", "co", "company", "nv", "n.v", "sa", "s.a", "ag",
    "gmbh", "kk", "pte", "pty", "bv", "ab", "as", "oy", "spa",
}
# tokens too generic to justify a merge candidate on their own
GENERIC_TOKENS = {
    "group", "holdings", "holding", "capital", "partners", "management",
    "energy", "digital", "data", "center", "centers", "centre", "centres",
    "infrastructure", "technologies", "technology", "tech", "global",
    "international", "ventures", "venture", "ai", "cloud", "computing",
    "the", "of", "and", "&", "new", "power", "systems", "platforms",
    "project", "campus", "site", "phase", "fund", "trust", "properties",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def norm_key(name: str) -> str:
    """Casefolded, punctuation-free, legal-suffix-stripped identity key.
    This is THE entity-identity normalization of the registry: two surfaces
    with the same norm_key form one provisional cluster."""
    s = _PUNCT_RE.sub(" ", str(name or "").replace("&", " and "))
    toks = [t for t in _WS_RE.split(s.lower().strip()) if t]
    core = [t for t in toks if t.rstrip(".") not in LEGAL_SUFFIX_TOKENS]
    return " ".join(core if core else toks)


def legal_form(name: str) -> Optional[str]:
    toks = [t.rstrip(".").lower() for t in _WS_RE.split(_PUNCT_RE.sub(" ", str(name or ""))) if t]
    for t in reversed(toks):
        if t in LEGAL_SUFFIX_TOKENS:
            return t
    return None


def trigrams(s: str) -> set:
    t = "  " + norm_key(s).replace(" ", " ") + "  "
    return {t[i:i + 3] for i in range(len(t) - 2)} if t.strip() else set()


def tri_jaccard(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class RetrievalIndex:
    """Inverted token index with IDF weights over cluster keys.

    Deterministic replacement for the sequential registry scan: candidates()
    is a pure function of the frozen key set, so resolution has no order
    dependence and never scans the full registry per lookup.
    """

    def __init__(self, keys: Iterable[str]):
        self.keys = sorted(set(keys))
        self.tok_index: dict[str, set] = defaultdict(set)
        self.df: Counter = Counter()
        # ENT-2: initials of every multi-token key ("canada pension plan
        # investment board" -> "cppib"), so an acronym query can meet its
        # expansion.  Acronyms share zero tokens and near-zero trigrams with
        # their expansion, so without this probe candidates() returns nothing
        # and the LLM judge is never consulted on exactly the pairs it would
        # get right.
        self.by_initials: dict[str, set] = defaultdict(set)
        for k in self.keys:
            toks = set(k.split())
            for t in toks:
                self.tok_index[t].add(k)
                self.df[t] += 1
            ordered = k.split()
            if len(ordered) >= 2:
                init = "".join(t[0] for t in ordered)
                if len(init) >= 2:
                    self.by_initials[init].add(k)
        self.n = max(1, len(self.keys))

    def _idf(self, tok: str) -> float:
        return math.log(1.0 + self.n / (1 + self.df.get(tok, 0)))

    def candidates(self, key: str, k: int = 5, exclude: Optional[set] = None) -> list[tuple[str, float]]:
        """Top-k (candidate_key, score); generic-only overlaps are filtered."""
        toks = [t for t in key.split() if t]
        pool: Counter = Counter()
        for t in toks:
            for other in self.tok_index.get(t, ()):
                if other == key or (exclude and other in exclude):
                    continue
                pool[other] += self._idf(t)
        scored = []
        for other, tok_score in pool.items():
            shared = set(key.split()) & set(other.split())
            if shared and shared <= GENERIC_TOKENS:
                continue                      # generic-only overlap: never a candidate
            tj = tri_jaccard(key, other)
            score = tok_score + 2.0 * tj
            if tj >= 0.34 or tok_score >= 1.2:
                scored.append((other, round(score, 4)))
        # ENT-2: acronym/initialism probe.  A short single-token query is
        # matched against candidate initials: exact ("cppib" == cppib), query
        # extends initials ("tsmc".startswith("tsm")), or initials extend the
        # query.  Admitted candidates route through the NORMAL path -- the LLM
        # judge still decides ("SAP" vs "Strategic Aluminum Partners" must be
        # judged with context, never auto-merged on initials).  Scored at the
        # token admission floor so genuine token matches outrank them.
        if " " not in key and 2 <= len(key) <= 6:
            have = {o for o, _ in scored}
            hits = set()
            for init, ks in self.by_initials.items():
                if (key == init
                        or (len(init) >= 3 and key.startswith(init))
                        or init.startswith(key)):
                    hits |= ks
            for other in sorted(hits):
                if other == key or (exclude and other in exclude) or other in have:
                    continue
                scored.append((other, round(1.2 + 2.0 * tri_jaccard(key, other), 4)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


# ---------------------------------------------------------------------------
# Amount buckets: fine = 2 significant digits, coarse = 1 significant digit.
# Coarse buckets drive candidate BLOCKING (recall-first); fine buckets only
# strengthen deterministic merge rules (precision-first).
# ---------------------------------------------------------------------------
def bucket_fine(v: float) -> str:
    return format(float(f"{float(v):.2g}"), ".12g")


def bucket_coarse(v: float) -> str:
    return format(float(f"{float(v):.1g}"), ".12g")


def representative_amount(a: dict) -> Optional[float]:
    """point, else high, else low (the baseline's representative rule)."""
    for k in ("amount_native_point", "amount_native_high", "amount_native_low"):
        v = coerce_num(a.get(k))
        if v is not None:
            return v
    return None


def amount_interval(a: dict) -> Optional[tuple[float, float]]:
    """Inclusive plausibility interval implied by the parsed bound."""
    p = coerce_num(a.get("amount_native_point"))
    lo = coerce_num(a.get("amount_native_low"))
    hi = coerce_num(a.get("amount_native_high"))
    bt = a.get("amount_bound_type") or "unclear"
    if bt == "range" and lo is not None and hi is not None:
        return (lo, hi)
    if bt == "max" and hi is not None:
        return (0.0, hi)
    if bt == "min" and lo is not None:
        return (lo, lo * 4.0)
    if p is not None:
        if bt == "approx":
            return (p * 0.9, p * 1.1)
        return (p, p)
    if hi is not None:
        return (0.0, hi)
    if lo is not None:
        return (lo, lo * 4.0)
    return None


def mw_value(row: dict) -> Optional[float]:
    qv = coerce_num(row.get("quantity_value"))
    unit = str(row.get("quantity_unit") or "").strip().lower()
    if qv is None or unit not in POWER_TO_MW:
        return None
    return qv * POWER_TO_MW[unit]


# ---------------------------------------------------------------------------
# Quarters
# ---------------------------------------------------------------------------
def q_of(iso: Any) -> Optional[str]:
    d = parse_date(iso)
    if d is None:
        return None
    y, q = TR.quarter_of_date(d.isoformat())
    return f"{y}Q{q}"


def q_bounds(label: str) -> tuple[str, str]:
    return TR.quarter_bounds(int(label[:4]), int(label[5]))


def q_range(labels: Iterable[str]) -> list[str]:
    labels = sorted(l for l in labels if l)
    if not labels:
        return []
    y, q = int(labels[0][:4]), int(labels[0][5])
    ye, qe = int(labels[-1][:4]), int(labels[-1][5])
    out = []
    while (y, q) <= (ye, qe):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


# ---------------------------------------------------------------------------
# Separateness cues for Fact identity: when the NEWER of two candidate
# observations carries one of these phrases, the pair is vetoed as two
# different Facts (fact_dedup Tier D).
# ---------------------------------------------------------------------------
SEPARATE_CUE_RE = re.compile(
    r"(?i)\b(?:in addition to|separate(?:ly)?\s+(?:from)?|a second|another\s+"
    r"(?:facility|financing|loan|agreement|commitment|round|deal|site|campus)|"
    r"additional\s+(?:facility|financing|loan|commitment)|a new,? separate)\b")


def git_head(root: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage manifest
# ---------------------------------------------------------------------------
class Manifest:
    """Per-stage provenance record; the success marker is written last.

    model_key/model_id record the registry adjudicator model configured for
    the run.  They are stamped on every stage manifest, including stages that
    make no LLM calls -- check the manifest's `llm` block to see whether the
    stage actually called the model."""

    def __init__(self, outdir: str, stage: str):
        self.outdir = outdir
        self.stage = stage
        self.t0 = time.time()
        self.data: dict = {"stage": stage, "run_tag": RUN_TAG,
                           "schema_version": SCHEMA_VERSION,
                           "code_commit": git_head(PIPELINE_ROOT),
                           "model_key": REG_MODEL,
                           "model_id": MODELS[REG_MODEL]["id"],
                           "inputs": {}, "outputs": {}, "stats": {},
                           "llm": {}, "success": False}

    def add_input(self, path: str) -> None:
        self.data["inputs"][os.path.basename(path)] = file_sha(path)

    def add_output(self, path: str, n: Optional[int] = None) -> None:
        self.data["outputs"][os.path.basename(path)] = {
            "sha": file_sha(path), "rows": n}

    def done(self, stats: dict, llm_stats: Optional[dict] = None) -> dict:
        self.data["stats"] = stats
        self.data["llm"] = llm_stats or {}
        self.data["duration_sec"] = round(time.time() - self.t0, 3)
        self.data["success"] = True
        path = os.path.join(self.outdir, "manifests", f"{self.stage}.json")
        write_json(self.data, path)
        return self.data


SOURCE_RANK = dict(SOURCE_RANK)          # re-export for stage modules
