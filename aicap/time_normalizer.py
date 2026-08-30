#!/usr/bin/env python3
"""Observation-level time extraction: gate -> read -> check -> resolve.

Given one observation (quote + temporal_context + doc_date + the resolved
target entity), produce zero or more TIME CLAIMS -- typed, dated, screened --
plus a per-observation review flag.  The 2026-08-19 revision replaced the
old T1-T4 architecture (rule hints -> LLM -> rule re-derivation -> universal
grading) after the claudetest run showed the model's calendar arithmetic was
essentially always right (91/92 envelope agreement, 14/14 weekdays) while
every serious defect was a RELEVANCE or ROLE judgment the old layers never
policed.  Division of labor now:

  GATE     deterministic routing only: skip the model when NO detector
           (scanner, deictic table, relative-cue regex) finds time evidence.
  READ     one LLM call per surviving row: roles / firmness / envelopes.
           The model owns the reading, including deictic arithmetic.
  CHECK    deterministic validation with three authority levels:
             * evidence integrity  -- verbatim / bracket / temporal-token /
               date-shape checks REJECT claims that carry no usable evidence;
             * knowledge injection -- fiscal labels resolve through the
               entity calendar (the ONLY envelope overwrite: knowledge the
               model cannot have; an unknown calendar keeps nulls);
             * screens             -- logical-impossibility and staleness
               flags (stale history, impossible tense, past deadline,
               arithmetic tripwire) that never overwrite -- they escalate.
  RESOLVE  one narrow LLM question per SCREENED claim (a few per hundred
           rows, not one grade per claim): does this expression date THIS
           event?  Verdicts act: not_this_event -> claim kept as evidence but
           dropped from the temporal projection (the fact falls back to
           disclosure_only, per the downgrade-never-block ruling);
           uncertain -> demoted; fix_envelope -> corrected dates.

Every date is an inclusive envelope [date_start, date_end]: "the event lies
in here".  A coarse expression keeps its coarse envelope -- no fake
precision.  `envelope_source` records which layer authored the dates
(llm | fiscal_calendar | deictic | escalation); `claim_status`
(active | demoted | dropped) is the single eligibility notion shared by the
temporal profile and fact dedup.

Driven by aicap.stages.time_extraction; shares the run's cache.sqlite with
the Judge-based stages under its own stage keys (s4n_time / s4n_esc).
"""
from __future__ import annotations
import datetime as dt
import re
from collections import Counter

from .llm import CallConfig, prompt_hash, sha16
from .config import (NORM_MODEL, fiscal_calendar_description,
                     fiscal_quarter_bounds, fiscal_year_bounds,
                     known_fiscal_start_month, model_id_of)
from . import time_rules as TR

# A realized claim whose envelope ENDS this many years (365-day years) before
# doc_date is suspicious of being a participant's own history (founding,
# listing, IPO) rather than this fact's event -- the screen escalates, never
# rejects: "December 11, 2015" nine years out is OpenAI's founding round and
# CORRECT.  Past-side only by design: long-dated FUTURE deadlines are normal
# in this corpus and no future analogue of the history class exists.
STALE_YEARS = 3

# ===========================================================================
# Claim schema vocabulary
# ===========================================================================
_RE_DATE_FULL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
# commit: announcement/commitment | bind: approval/signing/closing/effective |
# build: construction start / generic start | operate: operations begin |
# end: completion/termination/maturity/deadline (closes the valid interval) |
# period: accounting/reporting periods, as-of snapshots, payment windows.
_EVENT_ROLES = {"commit", "bind", "build", "operate", "end", "period"}
# realized: it happened | committed: a party-committed timetable/obligation |
# prospective: everything softer (expected/planned/targeted/estimated/...).
_FIRMNESS = {"realized", "committed", "prospective"}
_PRECISIONS = {"day", "month", "quarter", "year", "fiscal_year", "fiscal_quarter",
               "half_year", "multi_year", "unknown"}
# point: the event lies somewhere inside the envelope | period: the envelope IS
# the coverage span | deadline: "by X" -- date_end set, date_start null legal.
_INTERVALS = {"point", "period", "deadline"}
_LLM_RESOLUTION_METHODS = {"explicit", "doc_date_relative", "duration_derived",
                           "other_event_relative"}
_DURATION_UNITS = {"day", "week", "month", "quarter", "year"}
_CONFIDENCE = {"high", "medium", "low"}
_RE_FQ_LABEL = re.compile(r"^FY(\d{4})Q([1-4])$")

# A raw_expression that proposes DATES must contain at least one temporal
# token; "plans to invest" / "has announced" are status verbs, not time
# expressions, and the prompt's own ban proved unenforceable by prompt alone
# (T4 graded one such claim 5/5 on claudetest).  Dateless claims are exempt:
# "upon closing"-style other_event_relative claims legitimately carry none.
_TEMPORAL_TOKEN_RE = re.compile(
    r"\d"
    r"|january|february|march|april|may|june|july|august|september"
    r"|october|november|december"
    r"|\b(?:jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\bweds?\b"
    r"|today|yesterday|tomorrow|\bnow\b|immediately|already"
    r"|\bdays?\b|week|month|quarter|year|annual|fiscal|\bdate\b"
    r"|\bq[1-4]\b|\bh[12]\b|\bfy\b",
    re.I)


def _valid_date(s):
    if s is None:
        return True
    if not isinstance(s, str) or not _RE_DATE_FULL.match(s):
        return False
    try:
        dt.date.fromisoformat(s)
        return True
    except ValueError:
        return False


# ===========================================================================
# System prompts.  Their sha1 hashes version the cache namespaces, and the
# stage keys (s4n_time / s4n_esc) address the result caches -- never rename
# either without accepting a full re-bill of the stage.  2026-08-20: the two
# staged prompt edits LANDED (rule_candidates block dropped from prompt and
# call spec; participant-history rule added) ahead of the 2025 subsample
# rerun, where nothing is cached anyway; replaying an older frozen run now
# re-bills s4n_time -- see docs/TIME_LAYER_V2_IMPLEMENTATION_2026-08-19.md.
# ===========================================================================
TIME_NORM_SYS = """You are the AICapitalMap OBSERVATION TIME NORMALIZER v3. Extract zero or more TIME CLAIMS for ONE atomic capital observation. temporal_context is fact-local evidence proposed upstream; verbatim_quote is the source check. Judge RELEVANCE and ROLE yourself. Keep a claim only when its raw_expression is an exact span in temporal_context or verbatim_quote AND it modifies this observation, not another project, tranche, instrument, amount, or party in the quote.

For each claim, reason about START and END points: given only these words, what is the earliest and latest the time could be? Emit that inclusive ISO envelope (YYYY-MM-DD). A stated month/quarter/half-year/year expands to its first and last calendar day with matching granularity. NEVER compress a coarse expression to a fake day; never widen a precise one.

role -- one claim per distinct role:
  commit = the commitment/deal was announced or made ("announced", "committed", "plans to invest")
  bind   = it became legally binding (approval, signing, closing, effective date)
  build  = physical build begins (construction start, project start)
  operate= operations begin ("online", "operational", "energized")
  end    = the commitment's life ends (completion, termination, maturity, deadline)
  period = a coverage period: accounting/reporting period, "as of" snapshot, payment window
firmness: realized (it happened) | committed (an explicit party-committed timetable or obligation, e.g. "slated to open March 2027", a signed lease term) | prospective (expected/planned/targeted/estimated/conditional -- anticipated without a committed timetable). Grammatical tense alone licenses nothing.
interval: point (the event lies somewhere inside the envelope) | period (the envelope IS the validity/coverage span, e.g. a lease term, accounting period, as-of date) | deadline ("by / until / no later than X" -- date_end set, date_start null is CORRECT). "starting X onward" is interval=period with date_start set and date_end null.
duration_value/duration_unit: a stated tenor ("a 15-year lease" -> 15, "year") even when no start date is known; else null/null.
resolution_method: explicit | doc_date_relative | duration_derived | other_event_relative ("upon closing" -- keep dates null).

doc_date may resolve an explicit relative expression such as "last month" or "next year"; it is not itself an event. fiscal_period is document metadata and is not automatically a claim. deal_status and commitment_status help interpret role/firmness but never supply a date. Status words alone ("planned", "will invest", "Phase 1", "under construction") are not time expressions. A participant's OWN history -- its founding, listing, IPO, or an earlier unrelated deal -- is background, never this fact's event time.

FISCAL RULE: FY2025 / "fiscal 2025" is role=period: fiscal_year="FY2025", granularity=fiscal_year, dates null, resolution_method=explicit. A fiscal quarter ("Q3 FY2026") additionally sets fiscal_quarter_label="FY2026Q3" with granularity=fiscal_quarter and dates null. Deterministic code resolves dates from the entity's fiscal calendar afterward. Never create a period claim from document metadata alone; the expression must be fact-local source evidence.

raw_expression must be an exact source span. One expression may yield multiple claims (different roles). If evidence is ambiguous, keep dates null, lower confidence, and set time_needs_review=true rather than inventing.

Reply JSON ONLY:
{"events":[{"role":"commit|bind|build|operate|end|period","firmness":"realized|committed|prospective","date_start":"YYYY-MM-DD|null","date_end":"YYYY-MM-DD|null","granularity":"day|month|quarter|half_year|year|multi_year|fiscal_year|fiscal_quarter|unknown","interval":"point|period|deadline","raw_expression":"exact source span","resolution_method":"explicit|doc_date_relative|duration_derived|other_event_relative","confidence":"high|medium|low","fiscal_year":"FYyyyy|null","fiscal_quarter_label":"FYyyyyQn|null","duration_value":null,"duration_unit":"day|week|month|quarter|year|null"}],"time_needs_review":true|false}"""

TIME_ESCALATION_SYS = """You are the AICapitalMap TIME CLAIM RESOLVER. One TIME CLAIM extracted for ONE atomic capital observation was flagged by a deterministic screen (the screen text explains why). Decide what the flagged expression actually dates, using ONLY the quote, temporal_context, doc_date, and the claim shown -- not anything you know about the deal.

Verdicts:
  this_event     the expression dates THIS observation's capital event and the claim's envelope is right as stated
  fix_envelope   the expression dates THIS event but the envelope is wrong (wrong year anchor, wrong calendar frame, impossible tense) -- supply corrected date_start/date_end: inclusive ISO, coarse stays coarse, a deadline may leave date_start null
  not_this_event the expression dates something else: a participant's own history (founding, listing, IPO, an earlier unrelated deal), another project or tranche in the quote, or pure background
  uncertain      the evidence cannot settle it

Reply JSON ONLY: {"verdict":"this_event|fix_envelope|not_this_event|uncertain","date_start":"YYYY-MM-DD|null","date_end":"YYYY-MM-DD|null","reason":"<=20 words"}"""

NORM_TIME_VERSION = "s4n_time_" + prompt_hash(TIME_NORM_SYS)
NORM_ESC_VERSION = "s4n_esc_" + prompt_hash(TIME_ESCALATION_SYS)

# Screen names -> the sentence the RESOLVE prompt sees.
_SCREEN_TEXT = {
    "stale_history": (
        "the claim is REALIZED but its envelope ends more than "
        f"{STALE_YEARS} years before doc_date -- in this corpus that is "
        "often a participant's own history (founding, listing, IPO), not "
        "this capital event"),
    "impossible_tense": (
        "the claim is REALIZED but its envelope starts AFTER doc_date -- a "
        "completed event or reporting period cannot lie in the future; "
        "check the year anchor and the fiscal-vs-calendar frame"),
    "deadline_past": (
        "a committed/prospective DEADLINE lies entirely before doc_date -- "
        "likely a wrong year anchor; the intended deadline is normally the "
        "nearest FUTURE occurrence of the stated month/date"),
    "arith_mismatch": (
        "deterministic re-derivation of the expression disagrees with the "
        "claim's envelope"),
}


# --- validators for the LLM payloads ---------------------------------------
def _valid_time(o):
    """Validate the TIME NORMALIZER payload (role / firmness / interval)."""
    if not isinstance(o, dict) or not isinstance(o.get("events"), list):
        return False
    if not isinstance(o.get("time_needs_review"), bool):
        return False
    required = {"role", "firmness", "date_start", "date_end", "granularity",
                "interval", "raw_expression", "resolution_method", "confidence",
                "fiscal_year", "fiscal_quarter_label", "duration_value",
                "duration_unit"}
    for event in o["events"]:
        if not isinstance(event, dict) or not required <= set(event):
            return False
        if event.get("role") not in _EVENT_ROLES:
            return False
        if event.get("firmness") not in _FIRMNESS:
            return False
        if event.get("granularity") not in _PRECISIONS:
            return False
        if event.get("interval") not in _INTERVALS:
            return False
        if event.get("resolution_method") not in _LLM_RESOLUTION_METHODS:
            return False
        if event.get("confidence") not in _CONFIDENCE:
            return False
        if not isinstance(event.get("raw_expression"), str) or not event["raw_expression"].strip():
            return False
        if not _valid_date(event.get("date_start")) or not _valid_date(event.get("date_end")):
            return False
        fy = event.get("fiscal_year")
        if fy is not None and not (isinstance(fy, str) and re.fullmatch(r"FY\d{4}", fy)):
            return False
        fq = event.get("fiscal_quarter_label")
        if fq is not None and not (isinstance(fq, str) and _RE_FQ_LABEL.fullmatch(fq)):
            return False
        dv, du = event.get("duration_value"), event.get("duration_unit")
        if dv is not None and not (isinstance(dv, int) and 0 < dv <= 100):
            return False
        if du is not None and du not in _DURATION_UNITS:
            return False
        if (dv is None) != (du is None):
            return False
    return True


def _valid_escalation(o):
    """Validate the RESOLVE payload: a verdict, plus dates iff it fixes."""
    if not isinstance(o, dict):
        return False
    if o.get("verdict") not in {"this_event", "fix_envelope",
                                "not_this_event", "uncertain"}:
        return False
    ds, de = o.get("date_start"), o.get("date_end")
    if not _valid_date(ds) or not _valid_date(de):
        return False
    if o.get("verdict") == "fix_envelope":
        if ds is None and de is None:
            return False
        if ds and de and ds > de:
            return False
    return True


# ===========================================================================
# Mock LLM (offline wiring): deterministic canned responses, dispatched on a
# stable phrase in each system prompt.  The canned RESOLVE verdict is
# this_event (behavior-neutral), so a mock replay exercises the screens and
# wiring without inventing verdicts.
# ===========================================================================
def make_mock_fn():
    def mock_fn(system, user, cfg):
        if system.startswith("You are the AICapitalMap TIME CLAIM RESOLVER"):
            return ('{"verdict": "this_event", "date_start": null, '
                    '"date_end": null, "reason": "mock"}')
        if system.startswith("You are the AICapitalMap OBSERVATION TIME"):
            return ('{"events": [], "time_needs_review": false}')
        return "{}"
    return mock_fn


# ===========================================================================
# Claim identity.  The id signature reads the extractor's INLINE confidence
# (confidence_inline), so downstream re-scoring can never change a claim's
# identity.  envelope_source / screen_flags / claim_status /
# escalation_verdict / grounding_source are additive columns outside the
# signature for the same reason.  NOTE: envelope-CORRECTING layers (fiscal
# calendar, escalation fix_envelope) run before ids are minted, so a
# corrected claim gets a new content-addressed id by design.
# ===========================================================================
_ID_SIGNATURE_FIELDS = ("role", "firmness", "date_start", "date_end",
                        "granularity", "raw_expression", "resolution_method",
                        "confidence", "fiscal_year")


def _event_signature(event):
    values = []
    for key in _ID_SIGNATURE_FIELDS:
        if key == "confidence":
            values.append(event.get("confidence_inline", event.get("confidence")))
        else:
            values.append(event.get(key))
    return tuple(values)


class TimeNormalizer:
    """The gate/read/check/resolve pass over one run's observations.

    Shares the run's cache.sqlite (an aicap.llm.ResultCache) with the
    Judge-based stages, under its own stage keys (s4n_time / s4n_esc).
    Batched runs prefill the READ cache through the Message Batches API;
    RESOLVE calls are synchronous by design -- a handful per hundred rows."""

    def __init__(self, llm, cache, mock=False):
        self.llm = llm
        self.cache = cache
        self.model_id = model_id_of(NORM_MODEL)
        self.mock = mock
        # max_tokens=None -> the model maximum; fixed caps could truncate a
        # dense payload into quarantine.  Calls stream and stop_reason is
        # checked in aicap.llm.
        self.cfg = {kind: CallConfig(NORM_MODEL, max_tokens=None)
                    for kind in ("time", "esc")}
        self.llm_stats = {t: {"calls": 0, "hits": 0} for t in ("time", "esc")}
        # Cache keys whose quarantine was produced by THIS run's batch prefill
        # (the per-row path must not re-call them).
        self._batch_quarantined = set()
        self.time_tally = Counter()
        # Firms that acted as the fiscal entity for a time call but have NO
        # documented fiscal calendar -- the manual table-growth worklist.
        self.fiscal_unresolved = Counter()

    # ---- cached LLM call --------------------------------------------------
    def _cached_json(self, kind, cache_id, system, user, version, validate):
        key = self.cache.make_key(cache_id, f"s4n_{kind}", version, self.model_id)
        cc = self.cache.get(key)
        if cc is not None and cc.get("status") == "ok":
            self.llm_stats[kind]["hits"] += 1
            return cc["payload"], "ok"
        # A quarantine that THIS run's batch prefill already produced (incl.
        # its sync-fallback repair attempt) is final for this run -- re-calling
        # synchronously would double-bill the same failure.  A quarantine from
        # an EARLIER run still retries.
        if cc is not None and key in self._batch_quarantined:
            return None, "quarantine"
        obj, usage, status = self.llm.call_json(system, user, self.cfg[kind], validate=validate)
        self.cache.put(key, cache_id, f"s4n_{kind}", version, self.model_id, status, obj,
                       usage["in_tokens"], usage["out_tokens"],
                       stop_reason=usage.get("stop_reason"),
                       failure_reason=usage.get("failure_reason"), telemetry=usage)
        self.llm_stats[kind]["calls"] += 1
        return (obj if status == "ok" else None), status

    def _prefill_cache_batch(self, kind, version, requests, validate=None,
                             validators=None, label=""):
        """Push the UNCACHED calls of one pass through the Message Batches API
        (half token cost) and store the results in the ResultCache, so the
        subsequent per-row `_cached_json` path cache-hits.  `requests` =
        [(cache_id, system, user), ...] built by the same spec helpers the
        synchronous path uses -- prompts are byte-identical, so batch and sync
        runs share one cache.  Returns the number of items sent."""
        todo, keys = [], {}
        for cache_id, system, user in requests:
            key = self.cache.make_key(cache_id, f"s4n_{kind}", version, self.model_id)
            cc = self.cache.get(key)
            if cc is not None and (cc.get("status") == "ok"
                                   or key in self._batch_quarantined):
                continue
            keys[cache_id] = key
            todo.append((cache_id, system, user))
        if not todo:
            return 0

        def _persist(cache_id, res):
            # Write-through as each item completes: an interrupted wave then
            # loses only in-flight items, never paid-for finished ones.  The
            # bookkeeping loop below rewrites the same rows (idempotent).
            obj, usage, status = res
            self.cache.put(keys[cache_id], cache_id, f"s4n_{kind}", version,
                           self.model_id, status, obj,
                           usage["in_tokens"], usage["out_tokens"],
                           stop_reason=usage.get("stop_reason"),
                           failure_reason=usage.get("failure_reason"),
                           telemetry=usage)

        results = self.llm.call_json_batch(todo, self.cfg[kind], validate=validate,
                                           validators=validators, label=label,
                                           on_result=_persist)
        for cache_id, _system, _user in todo:
            obj, usage, status = results[cache_id]
            self.cache.put(keys[cache_id], cache_id, f"s4n_{kind}", version,
                           self.model_id, status, obj,
                           usage["in_tokens"], usage["out_tokens"],
                           stop_reason=usage.get("stop_reason"),
                           failure_reason=usage.get("failure_reason"),
                           telemetry=usage)
            self.llm_stats[kind]["calls"] += 1
            if status != "ok":
                self._batch_quarantined.add(keys[cache_id])
        return len(todo)

    # ============ the passes ============
    @staticmethod
    def _event_from_llm(proposed):
        """LLM payload -> claim row (canonical field names)."""
        return {
            "role": proposed.get("role"), "firmness": proposed.get("firmness"),
            "date_start": proposed.get("date_start"), "date_end": proposed.get("date_end"),
            "granularity": proposed.get("granularity"),
            "interval": proposed.get("interval"),
            "raw_expression": proposed.get("raw_expression"),
            "resolution_method": proposed.get("resolution_method"),
            "confidence": proposed.get("confidence"),
            "confidence_inline": proposed.get("confidence"),
            "fiscal_year": proposed.get("fiscal_year"),
            "fiscal_quarter_label": proposed.get("fiscal_quarter_label"),
            "duration_value": proposed.get("duration_value"),
            "duration_unit": proposed.get("duration_unit"),
            "calendar_quarter": None,
            # Audit-transparency dials, all OUTSIDE the id signature:
            # which layer authored the envelope, which screens fired, what
            # the resolver said, whether the fact-projection may use it, and
            # where the expression was found verbatim.
            "envelope_source": "llm",
            "screen_flags": [],
            "escalation_verdict": None,
            "claim_status": "active",
            "grounding_source": None,
        }

    def _fiscal_entity_key(self, row, fiscal_entity):
        """Entity-keyed fiscal resolution: the resolved TARGET's calendar
        wins; the chunk filer is the fallback. Returns (name, 'target'|'filer'|None)."""
        for name, tag in ((fiscal_entity, "target"), (row.get("firm"), "filer")):
            if name and known_fiscal_start_month(name) is not None:
                return name, tag
        return (fiscal_entity or row.get("firm")), None

    def _time_call_spec(self, row, fiscal_entity):
        """GATE + the exact READ request for one observation. Deterministic and
        side-effect-free (no tallies), so the batch prefill and the synchronous
        path can both call it and land on the same cache key. Returns a dict:
        skip=True means no temporal evidence by ANY detector (short-circuit);
        otherwise cache_id/user plus the artifacts normalize_time needs.
        2026-08-20: the rule_candidates block is GONE from the prompt (staged
        edit; hints never reached the deictic class and the model did not
        need them elsewhere) -- the deterministic scanner survives only as
        the GATE's evidence detector.  The gate itself is unchanged from the
        v2 revision (the old gate consulted the digit-bound scanner alone and
        silently dropped rows whose only evidence was a bare deictic: the
        $100B TSMC "last month" row on claudetest)."""
        tc = row.get("temporal_context")
        quote = row.get("verbatim_quote")
        tc_text = str(tc).strip() if tc else ""
        fkey, ftag = self._fiscal_entity_key(row, fiscal_entity)
        cand_q = TR.scan_time_candidates(quote)
        if (not tc_text and not cand_q and not TR.scan_deictic(quote)
                and not TR.RELATIVE_CUE_RE.search(str(quote or ""))):
            return {"skip": True}
        lines = [
            f'temporal_context: {tc}',
            f'verbatim_quote: {quote}',
            f'doc_date: {row.get("doc_date")}',
            f'fiscal_period: {row.get("fiscal_period")}',
            f'entity_for_fiscal_calendar: {fkey}',
            f'entity_fiscal_calendar_metadata: {fiscal_calendar_description(fkey)}',
            f'deal_status: {row.get("deal_status")}',
            f'commitment_status: {row.get("commitment_status")}']
        # A row whose Stage-3 audit flagged the inline time capture gets an
        # explicit warning: READ must re-derive every claim from the quote
        # instead of rubber-stamping temporal_context.
        if row.get("audit_time_bad"):
            lines.append("stage3_time_flag: the Stage-3 audit marked this observation's "
                         "inline time capture (temporal_context / fiscal_period) BAD. "
                         "Treat them as untrusted hints: keep only claims whose "
                         "raw_expression you verify in verbatim_quote or temporal_context, "
                         "and re-derive every envelope yourself.")
        user = "\n".join(lines)
        return {"skip": False, "cache_id": f'time|{row.get("raw_id")}|{sha16(user)}',
                "user": user, "tc_text": tc_text, "fkey": fkey, "ftag": ftag}

    # ---- RESOLVE: one narrow question per screened claim -------------------
    def _esc_call_spec(self, row, event, reasons):
        screen_lines = "; ".join(_SCREEN_TEXT[r] for r in reasons)
        user = "\n".join([
            f'screens: {screen_lines}',
            f'claim: role={event["role"]} | firmness={event["firmness"]} | '
            f'raw_expression="{event["raw_expression"]}" | '
            f'envelope={event["date_start"]}..{event["date_end"]} | '
            f'granularity={event["granularity"]} | interval={event["interval"]}',
            f'doc_date: {row.get("doc_date")}',
            f'temporal_context: {row.get("temporal_context")}',
            f'verbatim_quote: {row.get("verbatim_quote")}'])
        return f'esc|{row.get("raw_id")}|{sha16(user)}', user

    def _resolve_escalation(self, row, event):
        """Ask the resolver about one screened claim and APPLY the verdict.
        Quarantine fails safe: demote (never trust, never drop)."""
        cache_id, user = self._esc_call_spec(row, event, event["screen_flags"])
        obj, _status = self._cached_json("esc", cache_id, TIME_ESCALATION_SYS,
                                         user, NORM_ESC_VERSION, _valid_escalation)
        self.time_tally["escalated_claims"] += 1
        if obj is None:
            self.time_tally["esc_quarantine"] += 1
            event["claim_status"] = "demoted"
            event["escalation_verdict"] = "quarantine"
            return
        verdict = obj["verdict"]
        event["escalation_verdict"] = verdict
        self.time_tally[f"esc_{verdict}"] += 1
        if verdict == "not_this_event":
            event["claim_status"] = "dropped"
        elif verdict == "uncertain":
            event["claim_status"] = "demoted"
        elif verdict == "fix_envelope":
            event["date_start"] = obj.get("date_start")
            event["date_end"] = obj.get("date_end")
            event["envelope_source"] = "escalation"

    def normalize_time(self, row, fiscal_entity=None):
        dd = row.get("doc_date")
        dd_iso = str(dd)[:10] if dd and _valid_date(str(dd)[:10]) else None
        stale_cut = None
        if dd_iso:
            stale_cut = (dt.date.fromisoformat(dd_iso)
                         - dt.timedelta(days=365 * STALE_YEARS)).isoformat()
        spec = self._time_call_spec(row, fiscal_entity)

        # ---- GATE: deterministic routing only
        if spec["skip"]:
            self.time_tally["skipped_no_evidence"] += 1
            return False, []
        if not spec["tc_text"]:
            self.time_tally["quote_rescued"] += 1
        fkey, ftag = spec["fkey"], spec["ftag"]
        # ftag is None iff neither the resolved target nor the filer has a
        # documented fiscal calendar -- record the firm once per row for the
        # manual "top unresolved fiscal-calendar firms" report.
        if ftag is None and fkey:
            self.fiscal_unresolved[str(fkey)] += 1

        # ---- READ: one LLM call -- roles / firmness / envelopes
        obj, status = self._cached_json("time", spec["cache_id"], TIME_NORM_SYS,
                                        spec["user"], NORM_TIME_VERSION, _valid_time)
        self.time_tally["llm"] += 1
        if obj is None:
            self.time_tally["llm_quarantine"] += 1
            self.time_tally["needs_review"] += 1
            return True, []

        # ---- CHECK: evidence integrity, knowledge injection, screens
        review = bool(obj.get("time_needs_review"))
        quote_text = str(row.get("verbatim_quote") or "")
        quote_ws = re.sub(r"\s+", " ", quote_text)
        source = f'{row.get("temporal_context") or ""}\n{quote_text}'
        source_ws = re.sub(r"\s+", " ", source)
        valid_events = []
        for proposed in obj.get("events", []):
            event = self._event_from_llm(proposed)
            raw_expression = str(event.get("raw_expression") or "").strip()
            if not raw_expression:
                review = True
                self.time_tally["rejected_nonverbatim_event"] += 1
                continue
            # Bracket rule: the " [...] " join convention lives in QUOTES; a
            # bracketed insertion inside an expression is editorial text
            # unless the quote itself carries it verbatim (the
            # "by [mid-February]" channel on claudetest).
            if (("[" in raw_expression or "]" in raw_expression)
                    and raw_expression not in quote_text):
                review = True
                self.time_tally["rejected_bracketed_expression"] += 1
                continue
            if raw_expression not in source:
                if re.sub(r"\s+", " ", raw_expression) in source_ws:
                    self.time_tally["verbatim_ws_normalized"] += 1
                else:
                    review = True
                    self.time_tally["rejected_nonverbatim_event"] += 1
                    continue
            event["grounding_source"] = (
                "quote" if (raw_expression in quote_text
                            or re.sub(r"\s+", " ", raw_expression) in quote_ws)
                else "temporal_context")
            # Temporal-token gate: a claim that PROPOSES DATES from an
            # expression containing no temporal vocabulary at all is a status
            # verb wearing a costume ("plans to invest" -> [doc_date]).
            if (not _TEMPORAL_TOKEN_RE.search(raw_expression)
                    and (event.get("date_start") or event.get("date_end"))):
                review = True
                self.time_tally["rejected_no_temporal_token"] += 1
                continue
            start, end = event.get("date_start"), event.get("date_end")
            if not _valid_date(start) or not _valid_date(end):
                review = True
                self.time_tally["rejected_invalid_date"] += 1
                continue
            if start and end and start > end:
                review = True
                self.time_tally["rejected_reversed_interval"] += 1
                continue

            # loose scanner parse: fill-only backfills (labels, tenors)
            exp = TR.expand_expression(raw_expression)
            if exp and exp.get("fiscal_quarter") and not event.get("fiscal_quarter_label"):
                event["fiscal_quarter_label"] = f'{exp["fiscal_year"]}Q{exp["fiscal_quarter"]}'
            if exp and exp.get("fiscal_year") and not event.get("fiscal_year"):
                event["fiscal_year"] = exp["fiscal_year"]

            # FISCAL RULES (entity-keyed): an FY label forces role=period;
            # dates resolve via the entity calendar; unknown calendar keeps
            # nulls (never a silent January-year guess).  This is the ONLY
            # envelope overwrite left in the layer: knowledge injection.
            match = re.search(r"\bFY\s*'?(\d{2}\b|\d{4})", raw_expression, flags=re.I)
            fy_from_expr = None
            if match:
                y = match.group(1).strip()
                fy_from_expr = f"FY{2000 + int(y)}" if len(y) == 2 else f"FY{y}"
            if match and event["role"] != "period":
                event["role"] = "period"
                review = True
                self.time_tally["corrected_fy_role"] += 1
            if event["role"] == "period":
                fq_label = event.get("fiscal_quarter_label")
                fiscal_year = fy_from_expr or event.get("fiscal_year")
                if fq_label and _RE_FQ_LABEL.fullmatch(fq_label):
                    fy_num = int(_RE_FQ_LABEL.fullmatch(fq_label).group(1))
                    q_num = int(_RE_FQ_LABEL.fullmatch(fq_label).group(2))
                    event["fiscal_year"] = f"FY{fy_num}"
                    event["granularity"] = "fiscal_quarter"
                    bounds = fiscal_quarter_bounds(fkey, fy_num, q_num)
                    if bounds:
                        event["date_start"], event["date_end"] = bounds
                        event["resolution_method"] = "fiscal_calendar"
                        event["envelope_source"] = "fiscal_calendar"
                        if ftag:
                            self.time_tally[f"fiscal_key_{ftag}"] += 1
                    else:
                        event["date_start"] = event["date_end"] = None
                        event["resolution_method"] = "explicit"
                elif fiscal_year:
                    event["fiscal_year"] = fiscal_year
                    event["granularity"] = "fiscal_year"
                    bounds = fiscal_year_bounds(fkey, int(fiscal_year[2:]))
                    if bounds:
                        event["date_start"], event["date_end"] = bounds
                        event["resolution_method"] = "fiscal_calendar"
                        event["envelope_source"] = "fiscal_calendar"
                        if ftag:
                            self.time_tally[f"fiscal_key_{ftag}"] += 1
                    else:
                        event["date_start"] = event["date_end"] = None
                        event["resolution_method"] = "explicit"

            # Fiscal DEICTICS are the one deictic class code still touches --
            # calendar knowledge, not arithmetic supervision.  A known
            # calendar resolves "the prior fiscal year"; an unknown one nulls
            # a guess (never a silent January-year reading).  Every other
            # deictic keeps the model's envelope: on claudetest the old
            # re-derivation WIDENED four correct doc-clamped readings of
            # "Earlier this year" -- rules were the error source, not the
            # model.
            if exp is None:
                dhits = TR.scan_deictic(raw_expression)
                if len(dhits) == 1 and dhits[0]["deictic_unit"] == "fiscal_year":
                    fy_start = known_fiscal_start_month(fkey)
                    if fy_start is None:
                        if event.get("date_start") or event.get("date_end"):
                            event["date_start"] = event["date_end"] = None
                            review = True
                            self.time_tally["deictic_fiscal_unresolved"] += 1
                    else:
                        dexp = TR.expand_deictic(raw_expression, dd, fy_start)
                        if dexp:
                            want = (dexp["date_start"], dexp["date_end"])
                            if (event.get("date_start"), event.get("date_end")) != want:
                                event["date_start"], event["date_end"] = want
                                event["envelope_source"] = "deictic"
                                self.time_tally["corrected_deictic_envelope"] += 1
                            event["granularity"] = dexp["granularity"]
                            event["resolution_method"] = "doc_date_relative"
            if exp and exp.get("duration_value") and event.get("duration_value") is None:
                event["duration_value"] = exp["duration_value"]
                event["duration_unit"] = exp["duration_unit"]

            # One-sided shapes: a bare end is legal ONLY as a deadline
            # ("somewhere before X" IS deadline semantics), so coerce any
            # non-deadline interval, for ANY role -- "already energized"
            # (point) and "To date" (period) alike.  A bare start stays legal
            # only as an open-ended period; other shapes are info-flagged.
            ds, de = event.get("date_start"), event.get("date_end")
            if (ds is None) != (de is None):
                if de and event.get("interval") != "deadline":
                    event["interval"] = "deadline"
                    self.time_tally["coerced_end_only_deadline"] += 1
                iv = event.get("interval")
                if not ((de and iv == "deadline") or (ds and iv == "period")):
                    review = True
                    self.time_tally["one_sided_interval"] += 1

            # ---- screens: flag, never overwrite ---------------------------
            flags = []
            sexp = TR.expand_expression(raw_expression, strict=True)
            if (sexp and not sexp["needs_anchor"] and not sexp.get("fiscal_year")
                    and sexp["date_start"] and sexp["date_end"]
                    and event.get("date_start") and event.get("date_end")
                    and event.get("interval") != "deadline"
                    and (event["date_start"], event["date_end"])
                    != (sexp["date_start"], sexp["date_end"])):
                flags.append("arith_mismatch")
            if dd_iso:
                if (event.get("firmness") == "realized" and event.get("date_end")
                        and event["date_end"] < stale_cut):
                    flags.append("stale_history")
                if (event.get("firmness") == "realized" and event.get("date_start")
                        and event["date_start"] > dd_iso):
                    flags.append("impossible_tense")
                if (event.get("interval") == "deadline"
                        and event.get("firmness") in ("committed", "prospective")
                        and event.get("date_end") and event["date_end"] < dd_iso):
                    flags.append("deadline_past")
            for flag in flags:
                self.time_tally[f"screen_{flag}"] += 1
            if flags:
                event["screen_flags"] = flags
                # ---- RESOLVE: one narrow question, verdict applied
                self._resolve_escalation(row, event)

            event["calendar_quarter"] = TR.calendar_quarter_of(event.get("date_start"),
                                                               event.get("date_end"))
            valid_events.append(event)

        # ids are minted AFTER every envelope-touching layer (fiscal,
        # escalation fix) so a claim's id always names its final content.
        seen = set()
        deduped = []
        for event in valid_events:
            signature = _event_signature(event)
            if signature in seen:
                continue
            seen.add(signature)
            event["raw_id"] = str(row.get("raw_id"))
            event["temporal_event_id"] = "AICAP-T-" + sha16(event["raw_id"], *signature)
            deduped.append(event)
        valid_events = deduped

        # Intra-observation lifecycle ordering sanity (flags only):
        # realized build must not start after realized operations.
        def _first_realized_start(roles):
            starts = [e["date_start"] for e in valid_events
                      if e["role"] in roles and e.get("firmness") == "realized"
                      and e.get("date_start")]
            return min(starts) if starts else None

        con = _first_realized_start({"build"})
        op = _first_realized_start({"operate"})
        if con and op and op < con:
            review = True
            self.time_tally["ordering_violation"] += 1

        valid_events.sort(key=lambda event: (event.get("date_start") or "9999-99-99",
                                             event.get("role") or "",
                                             event["temporal_event_id"]))
        self.time_tally["events"] += len(valid_events)
        if review:
            self.time_tally["needs_review"] += 1
        return review, valid_events

    # ================= batch driver =================
    def prefill_time_batch(self, pairs, label="time"):
        """Batch-prefill the READ calls for [(row, fiscal_entity), ...]."""
        reqs = []
        for row, fiscal_entity in pairs:
            spec = self._time_call_spec(row, fiscal_entity)
            if not spec["skip"]:
                reqs.append((spec["cache_id"], TIME_NORM_SYS, spec["user"]))
        return self._prefill_cache_batch("time", NORM_TIME_VERSION, reqs,
                                         validate=_valid_time, label=label)
