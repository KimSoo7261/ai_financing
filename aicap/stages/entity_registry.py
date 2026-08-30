#!/usr/bin/env python3
"""Stage `resolve_entities_objects` -- sequential entity + typed-object registry.

One arrival is resolved at a time in deterministic disclosure order. Entity
mentions and complete object_name surfaces use exact-key, bounded retrieval,
conservative auto, then LLM tiers. Generic vocabulary is down-weighted rather
than removed because words such as campus/facility/phase are meaningful in
this namespace.

Object identity is typed: candidates of another object_type are never shown,
and exact/auto matches additionally require compatible parent and associated-
target context. One complete source expression (for example "Hyperion campus
Phase 1") remains one object surface; deterministic code never decomposes it
into invented project/campus/phase records. Parent references are lookup-only
and materialize only for explicit, type-compatible component_of assertions.

The v2 output contract is entities.jsonl / objects.jsonl /
entity_links.jsonl / object_links.jsonl / observations_resolved.jsonl.
Entity ids remain sequential per run; object ids are content-derived RO ids.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter

from .common import (GENERIC_TOKENS, LEGAL_SUFFIX_TOKENS, Manifest,
                     is_nullish, legal_form, norm_key, parse_name_list,
                     read_jsonl, sha16, split_semi, write_jsonl)
from ..config import OBJECT_PARENT_TYPES, OBJECT_TYPE_SET
from ..progress import Heartbeat
from .judge import Judge, content_id

MINT_SLOTS = {"target", "counterparty", "parent", "object"}

# ---------------------------------------------------------------------------
# ENT-1: deterministic registry alias table (unchanged from the batch module;
# see its git history for provenance).  Both sides pass through norm_key at
# import; values must be fixed points.
# ---------------------------------------------------------------------------
_RAW_REGISTRY_ALIASES = {
    # entity_alias_fix.csv
    "AMD": "Advanced Micro Devices",
    "CPPIB": "Canada Pension Plan Investment Board",
    "GIP": "Global Infrastructure Partners",
    "TSMC": "Taiwan Semiconductor Manufacturing",
    "NTT": "Nippon Telegraph and Telephone",
    "AEP": "American Electric Power",
    "Digital Bridge": "DigitalBridge",
    "Alpha Wave": "Alphawave",
    "Partners Group Holding AG": "Partners Group",
    "SharonAI": "Sharon AI",
    "First Energy": "FirstEnergy",
    # FISCAL_CALENDAR_ALIASES entries that are same-entity surfaces.
    "Amazon.com": "Amazon",
    "Facebook": "Meta",
    "Iris Energy": "IREN",
    "Taiwan Semiconductor": "Taiwan Semiconductor Manufacturing",
    "HPE": "Hewlett Packard Enterprise",
    "Supermicro": "Super Micro Computer",
    "Super Micro": "Super Micro Computer",
}
REGISTRY_ALIASES = {norm_key(a): norm_key(c)
                    for a, c in _RAW_REGISTRY_ALIASES.items()}
assert not (set(REGISTRY_ALIASES) & set(REGISTRY_ALIASES.values())), \
    "REGISTRY_ALIASES must be a fixed-point map (no canonical is an alias)"


def aliased_key(surface: str) -> str:
    """norm_key + the deterministic alias fold: THE entity identity key.
    Objects keep plain norm_key (the alias table is entity names only)."""
    k = norm_key(surface)
    return REGISTRY_ALIASES.get(k, k)


_PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:the\s+)?(?:jv|joint\s+venture|venture|consortium|company|group|"
    r"partnership|spv|vehicle|borrower|issuer|buyer|seller|lenders?|investors?|"
    r"parties|firm)$")

# ---------------------------------------------------------------------------
# Legal-form parsing + requirement-1 conflict veto (ported verbatim in
# semantics from pipeline/aicap_norm.py `_FORM_TAGS` / `_form_conflict_entity`).
# ---------------------------------------------------------------------------
_FORM_TAGS = {
    "inc": "corp", "incorporated": "corp", "corp": "corp", "corporation": "corp",
    "llc": "llc", "l.l.c": "llc", "lp": "lp", "l.p": "lp", "llp": "lp",
    "plc": "plc", "ltd": "ltd", "limited": "ltd", "gmbh": "gmbh", "ag": "ag",
}


def _form_of(name: str):
    for tok in re.findall(r"[a-z.]+", (name or "").lower()):
        f = _FORM_TAGS.get(tok.strip("."))
        if f:
            return f
    return None


def _form_conflict(surface: str, rec: dict) -> bool:
    """Mention has a legal form, the record has >=1, and none share family."""
    fa = _form_of(surface)
    if not fa:
        return False
    fams = {f for s in rec["aliases"] if (f := _form_of(s))}
    if not fams:
        return False
    return fa not in fams


# ---------------------------------------------------------------------------
# System prompts.  Body ported from the legacy normalizer (ENTITY_SYS_V2 /
# PROJECT_SYS in pipeline/aicap_norm.py) plus the ENT-2 acronym sentence.
# The literal phrase "REGISTRY JUDGE" + "choice" is what the orchestrator's
# --mock dispatcher keys on; keep it in both titles.
# ---------------------------------------------------------------------------
ENTITY_SYS = """You are the AICapitalMap ENTITY REGISTRY JUDGE (sequential normalizer v2). You decide whether ONE raw entity name (as written in a filing or news item) is the SAME real-world organization as one of a numbered list of EXISTING canonical registry entities. Obvious exact-key aliases are resolved upstream; you see the HARD residual. Each candidate line shows its canonical name, top aliases, legal forms seen, and mention count -- USE the legal form: a member 'X Inc.' and a vehicle 'X LLC' at the same base name are usually DISTINCT entities.

Return the INDEX of a candidate ONLY if it is the SAME operating organization -- an alias, short name, reordered/foreign legal name, or a brand denoting the same operating company (e.g. 'Open AI OpCo LLC' -> OpenAI; 'Cipher' -> Cipher Mining). An ACRONYM or initialism of a candidate's name (TSMC ~ Taiwan Semiconductor Manufacturing Company; CPPIB ~ Canada Pension Plan Investment Board) IS the same organization -- prefer its index. Do NOT merge a genuinely distinct SPV/JV vehicle into its parent or sponsor. A HOLDING/PARENT company and its OPERATING SUBSIDIARY are DISTINCT entities -- never merge them merely because they belong to the same corporate group, and never merge a physical site/campus name into the financing vehicle that funds it. When a candidate is plausibly the same operating company, PREFER its index; reserve -1 for an organization with no reasonable match.
Return -1 (new/distinct) for a real named organization that matches nothing listed. Return -2 (non_entity) ONLY when there is NO proper name at all -- a pure description ('a Singapore data-centre firm'), a placeholder ('the JV', 'the Venture'), bare junk ('GPUs'), or a bare country. A NAMED vehicle/SPV IS an entity -> its index or -1.

Reply JSON ONLY: {"choice": <index | -1 | -2>}"""

OBJECT_SYS = """You are the AICapitalMap TYPED OBJECT REGISTRY JUDGE. Decide whether ONE raw object name is the SAME real-world object as one numbered EXISTING canonical object. Every candidate has the SAME object_type as the mention; candidates of a different nature are excluded before this call. Preserve the complete source expression: a phase is not its campus, a building is not its facility, a power project is not the data center it serves, and an acquired company/business/asset portfolio is never a physical-project alias.

Return a candidate INDEX only for the SAME object (true alias, abbreviation, or wording variant). Use parent context and associated target names to disambiguate reused names. Return -1 for a genuine distinct object. Return -2 only when the string is not an object of the supplied type. Do not merge child into parent and do not infer ownership/service relationships as identity.

Reply JSON ONLY: {"choice": <index | -1 | -2>}"""


def _valid_choice(n_candidates: int):
    def check(o) -> bool:
        return (isinstance(o, dict) and isinstance(o.get("choice"), int)
                and -2 <= o["choice"] < n_candidates)
    return check


# ---------------------------------------------------------------------------
# Candidate-recall helpers (legacy `candidates()` scoring, requirement 2)
# ---------------------------------------------------------------------------
CAND_FLOOR = 0.15
_EPS = 1e-9
_GENERIC = GENERIC_TOKENS | LEGAL_SUFFIX_TOKENS

_TRI_MEMO: dict[str, frozenset] = {}


def _squeeze(key: str) -> str:
    return key.replace(" ", "")


def _sq_trigrams(sq: str) -> frozenset:
    hit = _TRI_MEMO.get(sq)
    if hit is None:
        if len(sq) < 3:
            hit = frozenset({sq} if sq else set())
        else:
            hit = frozenset(sq[i:i + 3] for i in range(len(sq) - 2))
        _TRI_MEMO[sq] = hit
    return hit


def _tri_jac(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# ENT-4: deterministic entity-kind axis (unchanged from the batch module).
# ---------------------------------------------------------------------------
_SOVEREIGN_KEYS = {norm_key(s) for s in (
    "Public Investment Fund", "PIF", "Mubadala", "Mubadala Investment Company",
    "MGX", "ADIA", "Abu Dhabi Investment Authority", "ADQ", "GIC",
    "Temasek", "Temasek Holdings", "QIA", "Qatar Investment Authority",
    "Khazanah", "Khazanah Nasional", "Norges Bank Investment Management")}
_SOVEREIGN_RE = re.compile(
    r"(?i)\bgovernment\b|\bministry\b|\btreasury\b|\bsovereign wealth\b|"
    r"\bstate of\b|\bdepartment of\b")
_GEO_KEYS = {norm_key(s) for s in (
    "United States", "US", "USA", "U.S.", "America", "China", "Japan",
    "South Korea", "Korea", "Taiwan", "India", "United Kingdom", "UK",
    "France", "Germany", "Saudi Arabia", "UAE", "United Arab Emirates",
    "Qatar", "Singapore", "Malaysia", "Indonesia", "Brazil", "Canada",
    "Mexico", "Australia", "Netherlands", "Ireland", "Norway", "Spain",
    "Italy", "Vietnam", "Thailand", "Philippines", "Israel", "Chile",
    "Europe", "Asia", "North America", "Middle East", "Latin America")}
_GROUP_DESC_RE = re.compile(
    r"(?i)-\s?backed\b|\bconsortium\b|\bgroup of\b|\bled by\b|"
    r"\binvestors?$|\b(?:sector|industry|players)\b")
_VEHICLE_RE = re.compile(
    r"(?i)\bspv\b|\bjv\b|\bholdco\b|\bholdings?\b|\bfunds?\b|\bvehicle\b|"
    r"\bjoint venture\b")


def _entity_kind(canonical: str, rec: dict) -> str:
    """entity_kind in {org, group_descriptor, geo, sovereign, vehicle}."""
    key = norm_key(canonical)
    if key in _SOVEREIGN_KEYS or _SOVEREIGN_RE.search(canonical):
        return "sovereign"
    if key in _GEO_KEYS:
        return "geo"
    if _GROUP_DESC_RE.search(canonical):
        return "group_descriptor"
    if _VEHICLE_RE.search(canonical):
        return "vehicle"
    inst = rec.get("instruments") or Counter()
    total = sum(inst.values())
    if (total >= 2 and inst.get("jv_spv", 0) >= 0.6 * total
            and legal_form(canonical) in {"llc", "lp", "llp"}):
        return "vehicle"
    return "org"


# ---------------------------------------------------------------------------
# The sequential registry pool (legacy Registry, per-run, no disk store)
# ---------------------------------------------------------------------------
class SeqRegistry:
    def __init__(self, kind: str, prefix: str, key_fn):
        self.kind = kind                     # "entity" | "object"
        self.prefix = prefix                 # "RE-E" | "RO"
        self.key_fn = key_fn                 # aliased_key | norm_key
        self.rows: dict[str, dict] = {}      # id -> record
        self.by_key: dict[object, set[str]] = {}  # typed identity key -> ids
        self._df: Counter = Counter()        # token -> #records containing it
        self._initials: dict[str, set] = {}  # ENT-2: initials -> {id}
        self._seq = 0
        self.links: list[dict] = []          # decision ledger rows
        self.n_renames = 0

    # ---- identity ----
    def key_of(self, surface: str) -> str:
        return self.key_fn(surface)

    # ---- token / IDF (legacy weights: generic tokens 0.1, else IDF) ----
    def _idf(self, token: str) -> float:
        n = len(self.rows)
        return math.log((n + 1) / (self._df.get(token, 0) + 1)) + 1.0

    def _weight(self, token: str) -> float:
        return 0.1 if token in _GENERIC else self._idf(token)

    def _lookup_key(self, key: str, object_type=None):
        return (object_type, key) if self.kind == "object" else key

    def _index_key(self, rec: dict, key: str) -> None:
        """Register one identity key on a record: by_key, token df, trigram
        string, initials probe."""
        if not key:
            return
        lookup = self._lookup_key(key, rec.get("object_type"))
        self.by_key.setdefault(lookup, set()).add(rec["id"])
        if key not in rec["keys"]:
            rec["keys"].add(key)
            for t in set(key.split()) - rec["_tok"]:
                self._df[t] += 1
            rec["_tok"] |= set(key.split())
            rec["_sq"].add(_squeeze(key))
            toks = key.split()
            if len(toks) >= 2:
                init = "".join(t[0] for t in toks)
                if len(init) >= 2:
                    self._initials.setdefault(init, set()).add(rec["id"])

    # ---- candidate recall (legacy scoring + ENT-2 initials probe) ----
    def candidates(self, surface: str, k: int = 5, object_type=None) -> list[tuple]:
        key = self.key_of(surface)
        q_toks = set(key.split())
        q_tri = _sq_trigrams(_squeeze(key))
        scored = []
        for eid, rec in self.rows.items():
            if self.kind == "object" and rec.get("object_type") != object_type:
                continue
            union = q_toks | rec["_tok"]
            shared = q_toks & rec["_tok"]
            wj = 0.0
            if union and shared:
                num = sum(self._weight(t) for t in shared)
                den = sum(self._weight(t) for t in union)
                wj = num / den if den else 0.0
            tri = max((_tri_jac(q_tri, _sq_trigrams(s)) for s in rec["_sq"]),
                      default=0.0)
            score = 0.7 * wj + 0.3 * tri
            if score + _EPS >= CAND_FLOOR:
                scored.append((round(score, 6), eid, rec))
        # ENT-2: a short single-token query is matched against candidate
        # initials so an acronym can meet its expansion; admitted at the
        # floor, the LLM still decides.
        if " " not in key and 2 <= len(key) <= 6:
            have = {eid for _s, eid, _r in scored}
            hits: set = set()
            for init, ids in self._initials.items():
                if (key == init
                        or (len(init) >= 3 and key.startswith(init))
                        or init.startswith(key)):
                    hits |= ids
            if self.kind == "object":
                hits = {eid for eid in hits
                        if self.rows[eid].get("object_type") == object_type}
            for eid in sorted(hits - have):
                scored.append((CAND_FLOOR, eid, self.rows[eid]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:k]

    # ---- requirement 4: canonical display name ----
    @staticmethod
    def _alias_rank(surface: str, meta: dict):
        multi = len(surface.split()) >= 2
        if meta.get("has_suffix") and meta.get("sec_n", 0) > 0:
            rank = 4
        elif meta.get("has_suffix"):
            rank = 3
        elif multi:
            rank = 2
        else:
            rank = 1
        return (rank, meta.get("n", 0), len(surface),
                tuple(-ord(c) for c in surface))

    def _best_canonical(self, rec: dict) -> str:
        if not rec["aliases"]:
            return rec["canonical"]
        best = max(rec["aliases"].items(),
                   key=lambda kv: self._alias_rank(kv[0], kv[1]))
        return best[0]

    def _recompute_canonical(self, rec: dict) -> bool:
        """v16.9 rename hysteresis: at most ONE rename per id per run, and
        only for a STRICTLY better alias score (ties keep the incumbent)."""
        if not rec["aliases"]:
            return False
        if rec["name_history"]:                    # already renamed this run
            return False
        incumbent = rec["canonical"]
        best = self._best_canonical(rec)
        if not best or best == incumbent:
            return False
        inc_meta = rec["aliases"].get(incumbent)
        if inc_meta is not None and \
                not (self._alias_rank(best, rec["aliases"][best])
                     > self._alias_rank(incumbent, inc_meta)):
            return False
        rec["name_history"].append({"old": incumbent, "new": best,
                                    "rule": "alias_score"})
        rec["canonical"] = best
        self.n_renames += 1
        self.links.append({"left_key": min(rec["keys"]) if rec["keys"] else None,
                           "right_key": None, "decision": "rename",
                           "decision_source": "rule:alias_score",
                           "evidence": f"{incumbent} -> {best}"[:300]})
        return True

    # ---- ingestion ----
    def create(self, surface: str, ctx: dict) -> str:
        self._seq += 1
        if self.kind == "object":
            seed = (ctx.get("object_type"), self.key_of(surface),
                    norm_key(ctx.get("parent_object_name") or ""),
                    tuple(sorted(ctx.get("target_ids") or ())),
                    ctx.get("raw_id"))
            eid = f"{self.prefix}-{sha16(*seed)}"
        else:
            eid = f"{self.prefix}{self._seq:04d}"
        rec = {"id": eid, "canonical": str(surface).strip(),
               "aliases": {}, "keys": set(), "slots": Counter(),
               "instruments": Counter(), "forms": Counter(),
               "n_mentions": 0, "first_doc": "9999", "name_history": [],
               "_tok": set(), "_sq": set()}
        if self.kind == "object":
            rec["object_type"] = ctx.get("object_type")
            rec["assoc_targets"] = []
            rec["parent_votes"] = Counter()
            rec["parent_surface_keys"] = set()
        self.rows[eid] = rec
        self.ingest_alias(eid, surface, ctx)
        return eid

    def ingest_alias(self, eid: str, surface: str, ctx: dict) -> None:
        """Count one mention of `surface` on record `eid`; register its key;
        re-vote the canonical (requirement 4)."""
        rec = self.rows[eid]
        surf = str(surface).strip()
        lf = legal_form(surf)
        al = rec["aliases"].setdefault(surf, {
            "n": 0, "sec_n": 0, "legal_form": lf,
            "has_suffix": lf is not None})
        al["n"] += 1
        if ctx.get("is_sec"):
            al["sec_n"] += 1
        rec["n_mentions"] += 1
        rec["slots"][ctx.get("slot") or "?"] += 1
        if ctx.get("instrument_class"):
            rec["instruments"][ctx["instrument_class"]] += 1
        if lf:
            rec["forms"][lf] += 1
        dd = str(ctx.get("doc_date") or "9999")
        rec["first_doc"] = min(rec["first_doc"], dd)
        self._index_key(rec, self.key_of(surf))
        if self.kind == "object":
            for tid in ctx.get("target_ids") or []:
                if tid and tid not in rec["assoc_targets"] \
                        and len(rec["assoc_targets"]) < 12:
                    rec["assoc_targets"].append(tid)
            pkey = norm_key(ctx.get("parent_object_name") or "")
            if pkey:
                rec["parent_surface_keys"].add(pkey)
        self._recompute_canonical(rec)


# ---------------------------------------------------------------------------
# The tier ladder (legacy _resolve_tiers, fitted to the Judge)
# ---------------------------------------------------------------------------
def _object_context_compatible(rec: dict, ctx: dict) -> bool:
    """High-precision exact-name guard for reused object names."""
    q_parent = norm_key(ctx.get("parent_object_name") or "")
    parents = rec.get("parent_surface_keys") or set()
    if q_parent and parents and q_parent not in parents:
        return False
    q_targets = {x for x in (ctx.get("target_ids") or []) if x}
    targets = set(rec.get("assoc_targets") or [])
    if q_targets and targets and not (q_targets & targets):
        return False
    return True


class Resolver:
    def __init__(self, judge: Judge, ereg: SeqRegistry, oreg: SeqRegistry):
        self.judge = judge
        self.regs = {"entity": ereg, "object": oreg}
        self.stage = {"entity": "r3_entity_v2", "object": "r3_object_v2"}
        self.system = {"entity": ENTITY_SYS, "object": OBJECT_SYS}
        self._cache: dict[tuple, str | None] = {}
        self.tally = {"entity": Counter(), "object": Counter()}

    def resolve(self, kind: str, surface, ctx: dict, create: bool = True):
        """Resolve ONE surface through the tiers (once per distinct surface
        per run).  Returns id | None."""
        reg = self.regs[kind]
        if surface is None or not str(surface).strip():
            return None
        surf = str(surface).strip()
        ckey = ((kind, ctx.get("object_type"), surf,
                 norm_key(ctx.get("parent_object_name") or ""),
                 tuple(sorted(ctx.get("target_ids") or ())))
                if kind == "object" else (kind, None, surf))
        if ckey in self._cache:
            eid = self._cache[ckey]
            if eid:
                reg.ingest_alias(eid, surf, ctx)     # still count the mention
            self.tally[kind]["cached"] += 1
            return eid
        eid, method = self._tiers(reg, surf, ctx, create)
        # legacy caching rule: lookup-only misses are NOT cached, so the same
        # surface later arriving in a minting slot can still create.
        if create or eid is not None or method == "non_entity":
            self._cache[ckey] = eid
        self.tally[kind][method] += 1
        return eid

    def _tiers(self, reg: SeqRegistry, surf: str, ctx: dict, create: bool):
        if _PLACEHOLDER_RE.match(surf):
            return None, "placeholder"
        key = reg.key_of(surf)
        if not key:
            return None, "non_entity"
        # Tier EXACT
        lookup = reg._lookup_key(key, ctx.get("object_type"))
        exact_ids = sorted(reg.by_key.get(lookup, set()))
        if len(exact_ids) == 1:
            eid = exact_ids[0]
            rec = reg.rows[eid]
            # Exact text is sufficient for entities. Objects additionally
            # require compatible parent/target context so reused site names do
            # not merge merely because their surface strings match.
            if reg.kind != "object" or _object_context_compatible(rec, ctx):
                reg.ingest_alias(eid, surf, ctx)
                return eid, "exact"
        cands = reg.candidates(surf, object_type=ctx.get("object_type"))
        if not cands:
            if not create:
                return None, "no_match_no_mint"
            return self._mint(reg, surf, ctx, "new_nocand")
        # Tier AUTO: unique squeezed-core-key equality, no form conflict
        sq = _squeeze(key)
        ck_eq = [c for c in cands
                 if sq in c[2]["_sq"] and not _form_conflict(surf, c[2])
                 and (reg.kind != "object"
                      or _object_context_compatible(c[2], ctx))]
        if len(ck_eq) == 1:
            eid = ck_eq[0][1]
            reg.links.append({"left_key": key,
                              "right_key": min(reg.rows[eid]["keys"]),
                              "decision": "merge",
                              "decision_source": "rule:auto_core_key",
                              "evidence": f"{key} == {sq}"[:300]})
            reg.ingest_alias(eid, surf, ctx)
            return eid, "auto"
        # Tier LLM
        return self._llm(reg, surf, key, cands, ctx, create)

    def _mint(self, reg: SeqRegistry, surf: str, ctx: dict, method: str):
        eid = reg.create(surf, ctx)
        reg.links.append({"left_key": reg.key_of(surf), "right_key": None,
                          "decision": "new", "decision_source": method,
                          "evidence": surf[:300]})
        return eid, method

    def _llm(self, reg: SeqRegistry, surf: str, key: str, cands: list,
             ctx: dict, create: bool):
        kind = reg.kind
        lines = []
        if kind == "entity":
            for i, (_s, _eid, rec) in enumerate(cands):
                aka = "; ".join(sorted(rec["aliases"],
                                       key=lambda a: -rec["aliases"][a]["n"])[:3])
                forms = ", ".join(sorted(rec["forms"])) or "-"
                lines.append(f"[{i}] {rec['canonical']} | aka: {aka or '-'} | "
                             f"legal: {forms} | mentions: {rec['n_mentions']}")
            mention = "\n".join([
                f'MENTION: "{surf}"',
                f'slot: {ctx.get("slot")}',
                f'instrument_class: {ctx.get("instrument_class")}',
                f'object: {ctx.get("object_name")}',
                f'other parties on row: {ctx.get("other_parties")}',
                f'summary: {ctx.get("summary")}'])
        else:
            ereg = self.regs["entity"]
            for i, (_s, _eid, rec) in enumerate(cands):
                aka = "; ".join(sorted(rec["aliases"],
                                       key=lambda a: -rec["aliases"][a]["n"])[:3])
                assoc = ", ".join(
                    ereg.rows[t]["canonical"]
                    for t in rec.get("assoc_targets", [])[:4]
                    if t in ereg.rows) or "-"
                parents = ", ".join(sorted(rec.get("parent_surface_keys") or [])) or "-"
                lines.append(f"[{i}] {rec['canonical']} | type: {rec.get('object_type')} | "
                             f"aka: {aka or '-'} | n: {rec['n_mentions']} | "
                             f"parents: {parents} | targets: {assoc}")
            mention = "\n".join([
                f'OBJECT MENTION: "{surf}"',
                f'object_type: {ctx.get("object_type")}',
                f'parent_object: {ctx.get("parent_object_name")}',
                f'instrument_class: {ctx.get("instrument_class")}',
                f'target(company): {ctx.get("target_canonicals")}',
                f'summary: {ctx.get("summary")}'])
        user = mention + "\n\nCANDIDATES:\n" + "\n".join(lines)
        ns = "ent" if kind == "entity" else "obj"
        item = content_id(ns, key, user)
        obj = self.judge.cached_json(self.stage[kind], item,
                                     self.system[kind], user,
                                     validate=_valid_choice(len(cands)))
        if obj is None:                        # quarantine -> conservative NEW
            reg.links.append({"left_key": key, "right_key": None,
                              "decision": "abstain", "decision_source": "llm",
                              "evidence": "quarantine"})
            if not create:
                return None, "no_match_no_mint"
            return self._mint(reg, surf, ctx, "llm_quarantine_new")
        ch = obj.get("choice", -1)
        if isinstance(ch, int) and 0 <= ch < len(cands):
            target = cands[ch]
            if kind == "entity" and _form_conflict(surf, target[2]):
                # requirement 1 veto
                reg.links.append({"left_key": key,
                                  "right_key": min(target[2]["keys"]),
                                  "decision": "different",
                                  "decision_source": "rule:form_conflict_veto",
                                  "evidence": f"form conflict vs {target[2]['canonical']}"[:300]})
                if not create:
                    return None, "no_match_no_mint"
                return self._mint(reg, surf, ctx, "form_veto_new")
            eid = target[1]
            reg.links.append({"left_key": key,
                              "right_key": min(target[2]["keys"]),
                              "decision": "merge", "decision_source": "llm",
                              "evidence": "judged same "
                              + ("organization" if kind == "entity" else "typed object")})
            reg.ingest_alias(eid, surf, ctx)
            return eid, "llm_match"
        if ch == -2:
            reg.links.append({"left_key": key, "right_key": None,
                              "decision": "non_entity",
                              "decision_source": "llm", "evidence": ""})
            return None, "non_entity"
        for _s, _eid, rec in cands:            # explicit negatives for audit
            reg.links.append({"left_key": key,
                              "right_key": min(rec["keys"]),
                              "decision": "different", "decision_source": "llm",
                              "evidence": "judged distinct"})
        if not create:
            return None, "no_match_no_mint"
        return self._mint(reg, surf, ctx, "llm_new")


# ---------------------------------------------------------------------------
def _export(reg: SeqRegistry, kind_fn=None) -> list[dict]:
    out = []
    for eid in sorted(reg.rows):
        rec = reg.rows[eid]
        aliases = sorted(rec["aliases"].items(),
                         key=lambda kv: (-kv[1]["n"], kv[0]))
        flags = []
        forms = {f for f in rec["forms"]
                 if f in {"inc", "corp", "corporation", "incorporated",
                          "llc", "lp", "llp"}}
        if {"llc", "lp", "llp"} & forms and \
                {"inc", "corp", "corporation", "incorporated"} & forms:
            flags.append("legal_form_conflict_within_cluster")
        row = {"id": eid, "canonical": rec["canonical"],
               "keys": sorted(rec["keys"]),
               "aliases": [a for a, _m in aliases[:8]],
               "n_mentions": rec["n_mentions"],
               "slots": dict(rec["slots"]),
               "first_doc_date": rec["first_doc"] if rec["first_doc"] != "9999" else None,
               "flags": flags,
               "name_history": rec["name_history"]}
        if kind_fn is not None:
            row["entity_kind"] = kind_fn(rec["canonical"], rec)
        if reg.kind == "object":
            row["object_type"] = rec.get("object_type")
            row["associated_target_ids"] = list(rec.get("assoc_targets", []))
            parent_votes = rec.get("parent_votes") or Counter()
            parent_ids = [pid for pid, _n in sorted(
                parent_votes.items(), key=lambda kv: (-kv[1], kv[0]))]
            row["parent_object_id"] = parent_ids[0] if parent_ids else None
            row["parent_object_ids_observed"] = parent_ids
            if len(parent_ids) > 1:
                row["flags"].append("multiple_parent_objects")
        out.append(row)
    return out


def run(outdir: str, judge: Judge) -> dict:
    man = Manifest(outdir, "resolve_entities_objects")
    obs_path = os.path.join(outdir, "observations.jsonl")
    man.add_input(obs_path)
    rows = read_jsonl(obs_path)

    ereg = SeqRegistry("entity", "RE-E", aliased_key)
    oreg = SeqRegistry("object", "RO", norm_key)
    res = Resolver(judge, ereg, oreg)

    # Arrival order: disclosure order (doc_date, raw_id) -- deterministic.
    ordered = sorted(rows, key=lambda r: (str(r.get("doc_date") or "9999"),
                                          str(r["raw_id"])))
    resolved_by_rid: dict[str, dict] = {}
    n_ent_mentions = n_object_mentions = 0
    hb = Heartbeat("registry", total=len(ordered))
    for _hb_done, row in enumerate(ordered):
        hb.beat(_hb_done, lambda: (
            f"ent {len(ereg.rows)} obj {len(oreg.rows)}"
            f" | ent tiers {dict(res.tally['entity'])}"))
        base = {"raw_id": row.get("raw_id"), "doc_date": row.get("doc_date"),
                "is_sec": str(row.get("source_type") or "").lower().startswith("sec"),
                "instrument_class": row.get("instrument_class"),
                "object_name": row.get("object_name"),
                "other_parties": row.get("other_parties"),
                "summary": str(row.get("summary") or "")[:300]}
        tgt_names = split_semi(row.get("target_firm"))
        tgt_ids = [res.resolve("entity", s, {**base, "slot": "target"})
                   for s in tgt_names]
        cp_ids = [res.resolve("entity", s, {**base, "slot": "counterparty"})
                  for s in split_semi(row.get("counterparty"))]
        par_ids = [res.resolve("entity", s, {**base, "slot": "parent"})
                   for s in parse_name_list(row.get("target_parent"))]
        oth_ids = [res.resolve("entity", s, {**base, "slot": "other_party"},
                               create=False)
                   for s in parse_name_list(row.get("other_parties"))]
        n_ent_mentions += (len(tgt_names) + len(cp_ids) + len(par_ids)
                           + len(oth_ids))
        oid = None
        oname = row.get("object_name")
        otype = row.get("object_type")
        if not is_nullish(oname) and otype in OBJECT_TYPE_SET:
            n_object_mentions += 1
            octx = {**base, "slot": "object", "object_type": otype,
                    "parent_object_name": row.get("parent_object_name"),
                    "target_ids": [t for t in tgt_ids if t],
                    "target_canonicals": "; ".join(
                        ereg.rows[t]["canonical"] for t in tgt_ids
                        if t and t in ereg.rows) or None}
            oid = res.resolve("object", oname, octx)
        resolved_by_rid[row["raw_id"]] = {
            "tgt": tgt_ids, "cp": cp_ids, "par": par_ids, "oth": oth_ids,
            "oid": oid, "parent_oid": None,
            "multi": bool(row.get("target_is_multi_entity")) or len(tgt_ids) > 1}

    # Parent references are lookup-only and run AFTER every primary object has
    # had a chance to mint. A parent string alone never invents a registry
    # object. Only type-compatible containment becomes a materialized edge.
    n_parent_resolved = n_parent_unresolved = n_parent_incompatible = 0
    for row in ordered:
        r = resolved_by_rid[row["raw_id"]]
        child_id = r.get("oid")
        pname = row.get("parent_object_name")
        ptype = row.get("parent_object_type")
        ctype = row.get("object_type")
        if not child_id or is_nullish(pname):
            continue
        if row.get("parent_object_relation") != "component_of":
            n_parent_unresolved += 1
            continue
        if ptype not in OBJECT_PARENT_TYPES.get(ctype, set()):
            n_parent_incompatible += 1
            continue
        pctx = {"raw_id": row.get("raw_id"), "doc_date": row.get("doc_date"),
                "is_sec": str(row.get("source_type") or "").lower().startswith("sec"),
                "slot": "parent_object", "object_type": ptype,
                "parent_object_name": None,
                "target_ids": [t for t in r.get("tgt", []) if t],
                "target_canonicals": None, "instrument_class": row.get("instrument_class"),
                "summary": str(row.get("summary") or "")[:300]}
        parent_id = res.resolve("object", pname, pctx, create=False)
        if parent_id and parent_id != child_id:
            r["parent_oid"] = parent_id
            oreg.rows[child_id]["parent_votes"][parent_id] += 1
            n_parent_resolved += 1
        else:
            n_parent_unresolved += 1

    # ---------- exports (final canonicals AFTER all renames) ----------
    entities = _export(ereg, kind_fn=_entity_kind)
    objects = _export(oreg)
    id_to_canon = {r["id"]: r["canonical"] for r in entities}
    oid_to_canon = {r["id"]: r["canonical"] for r in objects}

    resolved_rows = []
    for row in rows:                            # original file order
        r = resolved_by_rid[row["raw_id"]]
        resolved_rows.append({
            "raw_id": row["raw_id"],
            "target_entity_ids": r["tgt"],
            "target_canonicals": [id_to_canon.get(i) for i in r["tgt"]],
            "is_collective_target": r["multi"],
            "counterparty_entity_ids": r["cp"],
            "parent_entity_ids": r["par"],
            "other_party_entity_ids": r["oth"],
            "object_id": r["oid"],
            "object_canonical": oid_to_canon.get(r["oid"]),
            "object_type": row.get("object_type"),
            "parent_object_id": r["parent_oid"],
            "parent_object_canonical": oid_to_canon.get(r["parent_oid"]),
        })

    paths = {}
    paths["entities"] = os.path.join(outdir, "entities.jsonl")
    write_jsonl(entities, paths["entities"])
    paths["objects"] = os.path.join(outdir, "objects.jsonl")
    write_jsonl(objects, paths["objects"])
    paths["elinks"] = os.path.join(outdir, "entity_links.jsonl")
    write_jsonl(ereg.links, paths["elinks"])
    paths["olinks"] = os.path.join(outdir, "object_links.jsonl")
    write_jsonl(oreg.links, paths["olinks"])
    paths["resolved"] = os.path.join(outdir, "observations_resolved.jsonl")
    write_jsonl(resolved_rows, paths["resolved"])
    for p in paths.values():
        man.add_output(p)

    unresolved_targets = sum(1 for r in resolved_rows
                             if not any(r["target_entity_ids"]))
    stats = {"algorithm": "sequential_typed_objects_v2",
             "entity_mentions": n_ent_mentions,
             "entities": len(entities),
             "entity_tiers": dict(res.tally["entity"]),
             "entity_renames": ereg.n_renames,
              "object_mentions": n_object_mentions,
              "objects": len(objects),
              "object_tiers": dict(res.tally["object"]),
              "object_renames": oreg.n_renames,
              "parent_objects_resolved": n_parent_resolved,
              "parent_objects_unresolved": n_parent_unresolved,
              "parent_objects_incompatible": n_parent_incompatible,
             "rows_backfilled": len(resolved_rows),
             "rows_without_target_id": unresolved_targets}
    man.done(stats, judge.report())
    return stats
