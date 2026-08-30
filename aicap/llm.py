#!/usr/bin/env python3
"""LLM transport and result caching, shared by every pipeline stage.

Two critical pieces live here so no stage re-solves them:

  1. LLM         a thin, robust wrapper with two transports resolved per call
                 from the model key: AnthropicProvider (Messages API +
                 Message Batches) and OpenAICompatProvider (chat.completions
                 against the DashScope-compatible Qwen endpoints, realtime +
                 half-price Batch Chat).  The parse / repair / salvage /
                 quarantine orchestration is identical regardless of
                 transport, and a mock mode runs the whole pipeline with no
                 key.
  2. ResultCache a resumable sqlite cache keyed on
                 (chunk_id, stage, prompt_version, model_id): an interrupted
                 run continues where it stopped, editing a prompt invalidates
                 only that stage's namespace, and different models' verdicts
                 coexist without collision.

Plus the small JSONL/CSV helpers used everywhere.
Auth: ANTHROPIC_API_KEY for the anthropic provider; the env var named by the
model spec (DASHSCOPE_API_KEY) for openai_compat.  Requires: pip install
anthropic (always) and openai (only when a qwen model is selected).

NOTE on read_jsonl: this module's read_jsonl returns a GENERATOR (single
pass, no len()).  The registry stages use aicap.stages.common.read_jsonl,
which returns a list.  Never move a call between the two layers without
checking which one is in scope.
"""
from __future__ import annotations
import os, re, csv, json, time, hashlib, sqlite3, random
from dataclasses import dataclass, replace
from typing import Optional, Callable, Any, Iterable

from .config import MODELS, model_spec


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_jsonl(path: str) -> Iterable[dict]:
    """Yield one dict per line.  A GENERATOR -- see the module note."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(rows: list, columns: list, path: str) -> int:
    """Write `rows` (list of dicts) as CSV in the given column order.
    utf-8-sig so Windows Excel opens CJK/quoted text correctly."""
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c) for c in columns])
    return len(rows)


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def sha16(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Deterministic call config
# ---------------------------------------------------------------------------
@dataclass
class CallConfig:
    model_key: str
    # None -> the model's maximum output tokens (MODELS[...]["max_out"]).
    # Fixed caps silently truncate dense chunks, so "no output limit" is the
    # default and stop_reason is checked on every response.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    use_prompt_cache: bool = True
    max_retries: int = 6
    base_backoff: float = 2.0
    request_timeout: float = 600.0
    # openai_compat only: route this call to the half-price Batch Chat
    # endpoint (same request shape, queued up to an hour, answered on the
    # held connection).  Set by LLM._batch_chat_concurrent, never by stages.
    use_batch_endpoint: bool = False

    @property
    def model_id(self) -> str:
        return MODELS[self.model_key]["id"]

    @property
    def resolved_max_tokens(self) -> int:
        if self.max_tokens is not None:
            return self.max_tokens
        return MODELS[self.model_key].get("max_out", 8192)


# ---------------------------------------------------------------------------
# Resumable result cache  (one row per work unit)
# ---------------------------------------------------------------------------
class ResultCache:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        # WAL so a reader (progress inspection) never blocks the writer and
        # concurrent processes fail soft instead of "database is locked".
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results(
                key TEXT PRIMARY KEY,
                chunk_id TEXT, stage TEXT, prompt_version TEXT, model_id TEXT,
                status TEXT,          -- ok | partial | quarantine
                payload TEXT,         -- JSON for ok/partial; raw text for quarantine
                in_tokens INTEGER, out_tokens INTEGER, ts REAL,
                stop_reason TEXT, failure_reason TEXT, telemetry TEXT)""")
        # In-place, additive migration for caches created under v1.  Prompt and
        # schema versions keep incompatible successes unreachable; the ALTERs
        # preserve historical rows for audit instead of deleting them.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(results)")}
        for name in ("stop_reason", "failure_reason", "telemetry"):
            if name not in cols:
                self.conn.execute(f"ALTER TABLE results ADD COLUMN {name} TEXT")
        self.conn.commit()

    def close(self) -> None:
        """Release the SQLite handle; safe to call more than once."""
        conn, self.conn = getattr(self, "conn", None), None
        if conn is not None:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # A last-resort guard for historical call sites. Production runners
        # close explicitly so Windows does not retain cache.sqlite handles.
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def make_key(chunk_id, stage, prompt_version, model_id):
        return hashlib.sha1(f"{chunk_id}|{stage}|{prompt_version}|{model_id}"
                            .encode()).hexdigest()

    def get(self, key) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT status,payload,in_tokens,out_tokens,stop_reason,failure_reason,telemetry "
            "FROM results WHERE key=?",
            (key,)).fetchone()
        if not row:
            return None
        status, payload, it, ot, stop_reason, failure_reason, telemetry = row
        return {"status": status,
                "payload": json.loads(payload) if status in {"ok", "partial"} else payload,
                "in_tokens": it, "out_tokens": ot,
                "stop_reason": stop_reason, "failure_reason": failure_reason,
                "telemetry": json.loads(telemetry) if telemetry else {}}

    def put(self, key, chunk_id, stage, pv, model_id, status, payload, it, ot,
            *, stop_reason=None, failure_reason=None, telemetry=None):
        self.conn.execute(
            "INSERT OR REPLACE INTO results "
            "(key,chunk_id,stage,prompt_version,model_id,status,payload,in_tokens,"
            "out_tokens,ts,stop_reason,failure_reason,telemetry) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, chunk_id, stage, pv, model_id, status,
             json.dumps(payload) if status in {"ok", "partial"} else str(payload),
             it, ot, time.time(), stop_reason, failure_reason,
             json.dumps(telemetry or {}, ensure_ascii=False)))
        self.conn.commit()

    def payloads(self, stage: str, prompt_version: Optional[str] = None) -> list:
        """All 'ok' payloads for a stage. Pass prompt_version to restrict to ONE
        prompt revision -- ESSENTIAL when a cache.sqlite is carried across prompt
        versions (e.g. a copied folder), so observations from an older prompt do not
        contaminate the current run's output."""
        if prompt_version is None:
            rows = self.conn.execute(
                "SELECT payload FROM results WHERE stage=? AND status='ok'",
                (stage,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload FROM results WHERE stage=? AND status='ok' "
                "AND prompt_version=?", (stage, prompt_version)).fetchall()
        return [r[0] for r in rows]

    def stats(self):
        return self.conn.execute(
            "SELECT stage,status,COUNT(*),SUM(in_tokens),SUM(out_tokens) "
            "FROM results GROUP BY stage,status").fetchall()

    def failure_stats(self):
        """Compact actionable failure distribution for run telemetry."""
        return [
            {"stage": stage, "status": status,
             "failure_reason": reason or "none", "count": count}
            for stage, status, reason, count in self.conn.execute(
                "SELECT stage,status,COALESCE(failure_reason,''),COUNT(*) "
                "FROM results WHERE status!='ok' "
                "GROUP BY stage,status,COALESCE(failure_reason,'') "
                "ORDER BY stage,status,failure_reason"
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# JSON extraction + repair
# ---------------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


class JSONParseFailure(ValueError):
    """The response could not be decoded as JSON."""


class SchemaValidationFailure(ValueError):
    """Decoded JSON did not satisfy the stage's versioned contract."""


# Over-closed object recovery: in the extraction schema no observation-level
# value is an object except field_evidence, and field_evidence contains no
# nested objects -- so the byte sequence `}},"` (two closing braces directly
# followed by another key) can ONLY be an over-closed object.  Opus 5's one
# observed structural slip is exactly this: it closes field_evidence AND the
# observation (`"parent":null}},"summary":...`) then keeps writing keys.
# The rewrite is attempted only after normal parsing fails and is accepted
# only if the result parses -- it can never corrupt a valid payload.
_DOUBLE_CLOSE_RE = re.compile(r'\}\}(\s*,\s*")')


def extract_json(text: str):
    """Decode a model response as JSON, leniently before failing.

    Recovery ladder (each step only fires when the previous failed, and
    none can alter an already-valid payload):
      1. strict json.loads;
      2. strict=False -- admits raw control characters inside strings (the
         failure class behind ALL four structural quarantines in the
         flagship 2025 run);
      3. the `}},"` over-closed-object rewrite (see _DOUBLE_CLOSE_RE);
      4. the outermost {...}/[...] block, through the same ladder.
    Genuine syntax breakage still raises and takes the repair/quarantine
    path."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(t)
    except Exception:
        pass
    try:
        return json.loads(t, strict=False)
    except Exception:
        fixed = _DOUBLE_CLOSE_RE.sub(r'}\1', t)
        if fixed != t:
            try:
                return json.loads(fixed, strict=False)
            except Exception:
                pass
        m = _JSON_BLOCK.search(t)
        if m:
            block = m.group(0)
            try:
                return json.loads(block)
            except Exception:
                pass
            try:
                return json.loads(block, strict=False)
            except Exception:
                fixed = _DOUBLE_CLOSE_RE.sub(r'}\1', block)
                return json.loads(fixed, strict=False)
        raise


def salvage_observations(text: str) -> list:
    """Recover COMPLETE observation objects from a response cut off by
    max_tokens mid-JSON (lease-cost-heavy chunks pack many line items). Walk the
    `observations` array brace-by-brace, keep every top-level {...} that parses
    on its own, discard the truncated tail. Returns [] if nothing whole survives."""
    if not isinstance(text, str):
        return []
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    key = t.find('"observations"')
    if key == -1:
        return []
    lb = t.find("[", key)
    if lb == -1:
        return []
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i in range(lb + 1, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(t[start:i + 1]))
                except Exception:
                    pass
                start = None
        elif c == "]" and depth == 0:
            break
    return objs


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
# The transport is split out from LLM so the parse / salvage / repair /
# quarantine orchestration in LLM stays free of API details.  A provider
# exposes:
#   call(system, user, cfg) -> {text, in_tokens, out_tokens, stop_reason}
#       one request with the shared retry/backoff taxonomy;
#   supports_batch_api      True only where a real server-side batch API
#       exists (Anthropic); False routes call_json_batch through the
#       concurrent Batch Chat path instead;
#   run_batch_wave(...)     (Anthropic only) submit + poll + collect one
#       Message Batches wave.
def _map_finish_reason(fr) -> str:
    """Normalize an OpenAI finish_reason to the Anthropic-style stop_reason
    the rest of the pipeline reasons about (only "max_tokens" is
    load-bearing -- it drives the truncation/salvage path in call_json)."""
    if fr == "length":
        return "max_tokens"
    if fr == "stop":
        return "end_turn"
    return fr or "end_turn"


class AnthropicProvider:
    """Anthropic Messages API transport."""
    supports_batch_api = True

    # model ids that have 400'd on `temperature` -- shared across instances;
    # once a model is in here, subsequent calls omit the parameter.
    # Pre-seeded with the models whose API REMOVED sampling parameters
    # (Opus 5 / 4.8 / 4.7 reject temperature with a 400), so the first call
    # of a run never burns a rejected request discovering it.
    _NO_TEMPERATURE: set = {"claude-opus-5", "claude-opus-4-8",
                            "claude-opus-4-7"}

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic  (or run with mock=True)")
        self._client = anthropic.Anthropic()      # reads ANTHROPIC_API_KEY

    def call(self, system: str, user: str, cfg: "CallConfig") -> dict:
        import anthropic
        system_param = ([{"type": "text", "text": system,
                          "cache_control": {"type": "ephemeral"}}]
                        if cfg.use_prompt_cache else system)
        last_err = None
        for attempt in range(cfg.max_retries):
            try:
                # `temperature` is deprecated on newer models (e.g. Opus 4.7+,
                # which use adaptive thinking). Send it only when the config asks
                # for it AND this model has not already rejected it; the except
                # block below registers a model the first time it 400s on it.
                kwargs = dict(model=cfg.model_id, max_tokens=cfg.resolved_max_tokens,
                              system=system_param,
                              messages=[{"role": "user", "content": user}],
                              timeout=cfg.request_timeout)
                if cfg.temperature is not None and cfg.model_id not in type(self)._NO_TEMPERATURE:
                    kwargs["temperature"] = cfg.temperature
                # STREAM the response: model-max output budgets exceed what a
                # non-streaming request may carry, and streaming exposes
                # stop_reason so truncation is detected, never silent.
                with self._client.messages.stream(**kwargs) as stream:
                    resp = stream.get_final_message()
                text = "".join(b.text for b in resp.content if b.type == "text")
                return {"text": text, "in_tokens": resp.usage.input_tokens,
                        "out_tokens": resp.usage.output_tokens,
                        "stop_reason": resp.stop_reason}
            except (anthropic.RateLimitError, anthropic.APITimeoutError,
                    anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                last_err = e
                time.sleep(min(cfg.base_backoff * (2 ** attempt) + random.uniform(0, 1), 60))
            except anthropic.APIStatusError as e:
                if getattr(e, "status_code", None) in (529, 503, 502, 504):
                    last_err = e
                    time.sleep(min(cfg.base_backoff * (2 ** attempt) + random.uniform(0, 1), 60))
                elif (getattr(e, "status_code", None) == 400
                      and "temperature" in str(e).lower()
                      and cfg.model_id not in type(self)._NO_TEMPERATURE):
                    # this model deprecates `temperature` -- drop it and retry.
                    type(self)._NO_TEMPERATURE.add(cfg.model_id)
                    last_err = e
                    continue
                else:
                    last_err = e
                    break
            except Exception as e:
                last_err = e
                break
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def run_batch_wave(self, wave: list, cfg: "CallConfig", use_prompt_cache: bool,
                       poll_seconds: int, label: str, w0: int, total: int) -> tuple:
        """Submit ONE wave through the Message Batches API, poll to completion,
        and return (raw_results, missing):
          raw_results  ordered [(rid, raw|None)] in the API's result order --
                       raw = {text,in_tokens,out_tokens,stop_reason} for a
                       succeeded item, None for errored/canceled/expired;
          missing      rids whose result never came back (id_map order).
        The caller (LLM.call_json_batch) owns parse/salvage/sync-fallback, so
        this method is pure transport."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        id_map = {}
        batch_reqs = []
        for n, (rid, system, user) in enumerate(wave):
            cid = f"i{n}"
            id_map[cid] = (rid, system, user)
            system_param = ([{"type": "text", "text": system,
                              "cache_control": {"type": "ephemeral"}}]
                            if use_prompt_cache else system)
            batch_reqs.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=cfg.model_id, max_tokens=cfg.resolved_max_tokens,
                    system=system_param,
                    messages=[{"role": "user", "content": user}])))
        batch = self._client.messages.batches.create(requests=batch_reqs)
        print(f"  batch[{label}] items {w0 + 1}-{w0 + len(wave)} of {total}: "
              f"submitted {batch.id}")
        while True:
            batch = self._client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            rc = batch.request_counts
            print(f"  batch[{label}] {batch.id}: processing={rc.processing} "
                  f"succeeded={rc.succeeded} errored={rc.errored}")
            time.sleep(poll_seconds)
        raw_results, handled = [], set()
        for result in self._client.messages.batches.results(batch.id):
            rid, system, user = id_map[result.custom_id]
            handled.add(result.custom_id)
            if result.result.type == "succeeded":
                msg = result.result.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                raw_results.append((rid, {"text": text,
                                          "in_tokens": msg.usage.input_tokens,
                                          "out_tokens": msg.usage.output_tokens,
                                          "stop_reason": msg.stop_reason}))
            else:                              # errored | canceled | expired
                raw_results.append((rid, None))
        missing = [id_map[cid][0] for cid in id_map if cid not in handled]
        return raw_results, missing


class OpenAICompatProvider:
    """OpenAI chat.completions transport for the DashScope-compatible Qwen
    endpoints.  Two lazily-built clients: the realtime endpoint serves
    call_json; the Batch Chat endpoint (same request shape, half price,
    queued and answered on a held connection) serves calls whose CallConfig
    sets use_batch_endpoint.

    Thinking mode is explicitly DISABLED on every request: the qwen3 series
    defaults it ON, which bills reasoning tokens at output price and delays
    the JSON answer.  A model that rejects the parameter gets it dropped and
    the request retried (mirroring the temperature handling)."""
    supports_batch_api = False
    _clients: dict = {}      # (base_url, api_key_env) -> openai.OpenAI

    # Exception taxonomy keyed on the SDK's class NAME + HTTP status, so it
    # holds across SDK point releases.
    _RETRYABLE_NAMES = {"RateLimitError", "APITimeoutError", "APIConnectionError",
                        "APIConnectionTimeoutError", "InternalServerError"}
    _RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
    _FATAL_NAMES = {"AuthenticationError", "PermissionDeniedError", "NotFoundError",
                    "UnprocessableEntityError"}
    _FATAL_STATUS = {401, 403, 404, 422}

    def __init__(self, spec: dict):
        self.spec = spec

    def _client(self, batch: bool):
        """Lazily build (and cache) the client for the realtime or Batch Chat
        endpoint."""
        try:
            import openai
        except ImportError:
            raise SystemExit("pip install openai  (qwen models need the OpenAI SDK)")
        if batch:
            base_url = self.spec.get("batch_base_url")
        else:
            env = self.spec.get("base_url_env")
            base_url = (os.environ.get(env) if env else None) or self.spec.get("base_url")
        key_env = self.spec.get("api_key_env") or "OPENAI_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"{key_env} not set (required by model "
                             f"{self.spec.get('id')!r})")
        ckey = (base_url, key_env)
        client = type(self)._clients.get(ckey)
        if client is None:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            type(self)._clients[ckey] = client
        return client

    def _classify(self, e, want_json: bool, send_temp: bool,
                  send_thinking: bool) -> str:
        """Map a provider exception to one action:
          drop_json      400 rejecting response_format -> retry without it;
          drop_temp      400 rejecting temperature -> retry without it;
          drop_thinking  400 rejecting enable_thinking -> retry without it;
          retry          429 / timeout / 5xx / connection -> backoff + retry;
          fatal          auth / other 4xx / unknown -> stop (the validator
                         loop upstream turns a schema miss into a quarantine,
                         never a spin)."""
        name = type(e).__name__
        msg = str(e).lower()
        sc = getattr(e, "status_code", None)
        if sc == 400 or name == "BadRequestError":
            if want_json and ("response_format" in msg or "json" in msg):
                return "drop_json"
            if send_temp and "temperature" in msg:
                return "drop_temp"
            if send_thinking and "thinking" in msg:
                return "drop_thinking"
            return "fatal"
        if name in self._FATAL_NAMES or sc in self._FATAL_STATUS:
            return "fatal"
        if name in self._RETRYABLE_NAMES or sc in self._RETRYABLE_STATUS:
            return "retry"
        return "fatal"

    def call(self, system: str, user: str, cfg: "CallConfig") -> dict:
        client = self._client(cfg.use_batch_endpoint)
        want_json = True                      # ask for json_object; drop on reject
        send_temp = cfg.temperature is not None
        send_thinking = True
        last_err = None
        for attempt in range(cfg.max_retries):
            try:
                kwargs = dict(model=cfg.model_id, max_tokens=cfg.resolved_max_tokens,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                              timeout=cfg.request_timeout)
                if send_temp:
                    kwargs["temperature"] = cfg.temperature
                if want_json:
                    kwargs["response_format"] = {"type": "json_object"}
                if send_thinking:
                    kwargs["extra_body"] = {"enable_thinking": False}
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = getattr(resp, "usage", None)
                return {"text": text,
                        "in_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "out_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "stop_reason": _map_finish_reason(getattr(choice, "finish_reason", None))}
            except Exception as e:
                action = self._classify(e, want_json, send_temp, send_thinking)
                last_err = e
                if action == "drop_json":
                    want_json = False
                    continue
                if action == "drop_temp":
                    send_temp = False
                    continue
                if action == "drop_thinking":
                    send_thinking = False
                    continue
                if action == "retry":
                    time.sleep(min(cfg.base_backoff * (2 ** attempt) + random.uniform(0, 1), 60))
                    continue
                break                         # fatal
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# The client wrapper
# ---------------------------------------------------------------------------
class LLM:
    """Client wrapper with the stability guarantees: JSON parse, one repair
    retry, truncation salvage, per-item quarantine, and a transport circuit
    breaker.  In mock mode it returns a deterministic canned response (no
    network, no keys)."""

    # This many CONSECUTIVE transport failures aborts the run -- an API outage
    # must not silently quarantine a whole corpus one item at a time.
    TRANSPORT_FAIL_LIMIT = 8

    def __init__(self, mock: bool = False, mock_fn: Optional[Callable] = None):
        self.mock = mock
        self.mock_fn = mock_fn
        self._providers: dict = {}            # (provider, api_key_env) -> transport
        self._transport_failures = 0          # consecutive; reset on any success

    def _provider(self, cfg: CallConfig):
        """Resolve (and cache) the transport for a call's model key."""
        spec = model_spec(cfg.model_key)
        provider = spec["provider"]
        ckey = (provider, spec.get("api_key_env"))
        inst = self._providers.get(ckey)
        if inst is None:
            if provider == "anthropic":
                inst = AnthropicProvider()
            elif provider == "openai_compat":
                inst = OpenAICompatProvider(spec)
            else:
                raise SystemExit(f"unknown provider {provider!r} for model "
                                 f"{cfg.model_key!r}")
            self._providers[ckey] = inst
        return inst

    def call(self, system: str, user: str, cfg: CallConfig) -> dict:
        if self.mock:
            text = self.mock_fn(system, user, cfg) if self.mock_fn else "{}"
            return {"text": text, "in_tokens": len(system + user) // 4,
                    "out_tokens": len(text) // 4, "stop_reason": "end_turn"}
        return self._provider(cfg).call(system, user, cfg)

    @staticmethod
    def _parse_valid(text: str, validate) -> Any:
        try:
            obj = extract_json(text)
        except Exception as exc:
            detail = str(exc)
            # JSONDecodeError carries the parsed document: quote the exact
            # breakage site so the repair call knows WHAT to fix (a generic
            # "not valid JSON" message left the model re-making the same
            # mistake -- observed 10/12 repair failures on pilot wave 1).
            doc = getattr(exc, "doc", None)
            pos = getattr(exc, "pos", None)
            if isinstance(doc, str) and isinstance(pos, int):
                lo, hi = max(0, pos - 90), min(len(doc), pos + 40)
                detail += f" | text around the error: ...{doc[lo:hi]}..."
            raise JSONParseFailure(detail) from exc
        if validate and not validate(obj):
            # Validators may expose .explain(obj) naming the failing rule;
            # the repair call forwards it verbatim.
            explain = getattr(validate, "explain", None)
            detail = ""
            if explain is not None:
                try:
                    detail = str(explain(obj) or "")[:700]
                except Exception:
                    detail = ""
            raise SchemaValidationFailure(detail or "schema validation failed")
        return obj

    @staticmethod
    def _salvage(texts, validate):
        """Best-effort recovery of complete observation objects from truncated
        text. Returns (payload | None, n_salvaged). Every salvaged object is
        tagged `_truncated_salvage` so callers can COUNT the loss -- a salvage
        is never silent (run stats record it)."""
        best = []
        for cand in texts:
            if not isinstance(cand, str):
                continue
            rec = salvage_observations(cand)
            if len(rec) > len(best):
                best = rec
        if not best:
            return None, 0
        complete = {"observations": [dict(o) for o in best
                                     if isinstance(o, dict) and o.get("verbatim_quote")]}
        if complete["observations"] and (not validate or validate(complete)):
            for o in complete["observations"]:
                o["_truncated_salvage"] = True
            return complete, len(complete["observations"])
        return None, 0

    def call_json(self, system: str, user: str, cfg: CallConfig,
                  validate: Optional[Callable[[Any], bool]] = None):
        """Call, parse JSON, repair once if needed.
        Returns (obj, usage, status); status in {'ok','partial','quarantine'}.
        usage carries `stop_reason` and, when salvage fired, `salvaged` (row
        count) + `truncated` -- callers surface these in run stats.
        A transport failure that survived the provider's own retries becomes a
        PER-ITEM quarantine (payload `TRANSPORT_ERROR:`) instead of killing
        the run -- it self-heals on the next run via the earlier-run-
        quarantine retry contract -- unless the circuit breaker trips on
        consecutive failures (the API is down: fail fast)."""
        try:
            r = self.call(system, user, cfg)
        except Exception as err:
            # Batch Chat queue timeouts/disconnects are EXPECTED (and
            # unbilled); they fall back to a realtime call per item, so they
            # must not trip the run-abort breaker.  The realtime path keeps
            # full breaker protection.
            if not cfg.use_batch_endpoint:
                self._transport_failures += 1
                if self._transport_failures >= self.TRANSPORT_FAIL_LIMIT:
                    raise RuntimeError(
                        f"transport circuit breaker: {self._transport_failures} "
                        f"consecutive failures (last: {err})") from err
            usage = {"in_tokens": 0, "out_tokens": 0,
                     "stop_reason": "transport_error",
                     "failure_reason": "transport_error"}
            return f"TRANSPORT_ERROR: {err}", usage, "quarantine"
        self._transport_failures = 0
        usage = {"in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
                 "stop_reason": r.get("stop_reason")}
        truncated = r.get("stop_reason") == "max_tokens"
        if truncated:
            usage["truncated"] = True
        try:
            obj = self._parse_valid(r["text"], validate)
            if truncated:
                # Syntactically complete output can still be semantically
                # incomplete when the provider stopped at the token ceiling.
                usage["failure_reason"] = "max_tokens"
                return obj, usage, "partial"
            return obj, usage, "ok"
        except Exception as first_err:
            first_reason = ("json_parse_failure" if isinstance(first_err, JSONParseFailure)
                            else "schema_validation_failure")
            r2 = None
            if not truncated:
                # Malformed (not truncated) -> one repair retry. A response cut
                # at the model-max cap cannot be fixed by re-asking at the same
                # cap, so truncation skips straight to salvage.  The retry is
                # told exactly WHAT was invalid (parse position + context, or
                # the named schema rule) -- a generic message made the model
                # repeat the identical mistake.
                repair_user = (user + "\n\nYour previous output was INVALID: "
                               + str(first_err)[:800]
                               + "\nFix exactly this problem and re-emit the "
                                 "COMPLETE corrected JSON. ONLY the JSON, no "
                                 "prose, no code fences.")
                try:
                    r2 = self.call(system, repair_user, cfg)
                    usage["in_tokens"] += r2["in_tokens"]
                    usage["out_tokens"] += r2["out_tokens"]
                    obj = self._parse_valid(r2["text"], validate)
                    if r2.get("stop_reason") == "max_tokens":
                        usage["truncated"] = True
                        usage["stop_reason"] = "max_tokens"
                        usage["failure_reason"] = "max_tokens"
                        return obj, usage, "partial"
                    usage["repair_applied"] = True
                    return obj, usage, "ok"
                except Exception as second_err:
                    second_reason = ("json_parse_failure"
                                     if isinstance(second_err, JSONParseFailure)
                                     else "schema_validation_failure")
                    usage["failure_reason"] = f"repair_failed:{first_reason}:{second_reason}"
                    return r["text"], usage, "quarantine"
            # Salvage is permitted ONLY for a provider-confirmed token cutoff,
            # and is explicitly partial.  Format/schema failures never masquerade
            # as truncation and never become successful cache entries.
            salvaged, n = self._salvage((r["text"],), validate)
            usage["failure_reason"] = "max_tokens"
            if salvaged is not None:
                usage["salvaged"] = n
                return salvaged, usage, "partial"
            return {"_partial_raw": r["text"]}, usage, "partial"

    def call_json_batch(self, requests: list, cfg: CallConfig,
                        validate: Optional[Callable[[Any], bool]] = None,
                        poll_seconds: int = 30, wave_size: int = 500,
                        label: str = "",
                        validators: Optional[dict] = None,
                        on_result: Optional[Callable] = None) -> dict:
        """Run many INDEPENDENT JSON calls through the Message Batches API
        (half the token cost of the synchronous API). `requests` = [(req_id,
        system, user), ...]; returns {req_id: (obj, usage, status)} with the
        same per-item semantics as call_json: parse -> salvage (counted, never
        silent) -> sync fallback (which retains the repair retry) ->
        quarantine. Batch-side failures (errored / canceled / expired /
        missing) always fall back to a synchronous call, so no work item is
        ever lost to transport problems.

        `validators` maps req_id -> validator for stages whose schema depends
        on the request (the time grader requires every claim index of THAT
        observation graded exactly once); it overrides `validate` per item.

        `on_result(req_id, (obj, usage, status))` is invoked in the CALLING
        thread as each item completes, so callers can persist results
        incrementally -- an interrupted wave then loses only in-flight items,
        never completed ones.

        `temperature` is never sent on batch requests (the pinned models
        reject sampling parameters)."""
        def _v(rid):
            return (validators or {}).get(rid, validate)

        if self.mock or not requests:
            out: dict = {}
            for rid, system, user in requests:
                out[rid] = self.call_json(system, user, cfg, validate=_v(rid))
                if on_result:
                    on_result(rid, out[rid])
            return out

        provider = self._provider(cfg)
        if not provider.supports_batch_api:
            return self._batch_chat_concurrent(requests, cfg, _v, label, on_result)

        out: dict = {}

        def _emit(rid, res):
            out[rid] = res
            if on_result:
                on_result(rid, res)
        todo = list(requests)
        for w0 in range(0, len(todo), wave_size):
            wave = todo[w0:w0 + wave_size]
            wmap = {rid: (rid, system, user) for rid, system, user in wave}
            raw_results, missing = provider.run_batch_wave(
                wave, cfg, cfg.use_prompt_cache, poll_seconds, label, w0, len(todo))
            fallback = []
            for rid, raw in raw_results:
                if raw is None:                        # errored | canceled | expired
                    fallback.append(wmap[rid])
                    continue
                usage = {"in_tokens": raw["in_tokens"], "out_tokens": raw["out_tokens"],
                          "stop_reason": raw["stop_reason"], "batched": True}
                if raw["stop_reason"] == "max_tokens":
                    usage["truncated"] = True
                    usage["failure_reason"] = "max_tokens"
                    try:
                        obj = self._parse_valid(raw["text"], _v(rid))
                        _emit(rid, (obj, usage, "partial"))
                        continue
                    except Exception:
                        salvaged, ns = self._salvage((raw["text"],), _v(rid))
                        if salvaged is not None:
                            usage["salvaged"] = ns
                            _emit(rid, (salvaged, usage, "partial"))
                        else:
                            _emit(rid, ({"_partial_raw": raw["text"]}, usage, "partial"))
                        continue
                try:
                    _emit(rid, (self._parse_valid(raw["text"], _v(rid)), usage, "ok"))
                    continue
                except Exception:
                    # Non-truncation format/schema failure gets the full
                    # synchronous repair/rerun path; it is never salvaged.
                    fallback.append(wmap[rid])
            for rid in missing:                        # result never returned -> sync
                fallback.append(wmap[rid])
            if fallback:
                print(f"  batch[{label}]: {len(fallback)} item(s) -> synchronous fallback")
            for rid, system, user in fallback:
                _emit(rid, self.call_json(system, user, cfg, validate=_v(rid)))
        return out

    def _batch_chat_concurrent(self, requests: list, cfg: CallConfig,
                               vfn: Callable, label: str,
                               on_result: Optional[Callable] = None) -> dict:
        """Batch path for providers without a server-side batch API: run each
        item's call_json against the half-price Batch Chat endpoint from a
        bounded thread pool.  Requests queue server-side (up to an hour) and
        answer on the held connection; a failed or timed-out batch attempt
        (unbilled) falls back to ONE realtime call, so no work item is ever
        lost.  Per-item parse -> repair -> salvage -> quarantine semantics
        are unchanged, and the returned {req_id: (obj, usage, status)} is
        shape-identical to the Anthropic batch path.

        Concurrency defaults to 32 workers (AICAP_BATCH_CONCURRENCY); the
        service allows far more pending requests than any sane pool size."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        reqs = list(requests)
        workers = max(1, min(int(os.environ.get("AICAP_BATCH_CONCURRENCY", "32")),
                             len(reqs)))
        # Batch attempts wait out the queue (full hour) but do NOT loop the
        # normal 6-retry schedule -- a congested queue must fail over to
        # realtime, not block for six hours.  max_tokens is capped far above
        # any real answer in this pipeline: some gateways reject the model's
        # full output ceiling in batch mode, and the truncation/salvage
        # backstop still guards the cap.
        batch_cfg = replace(cfg, use_batch_endpoint=True,
                            request_timeout=3600.0, max_retries=2,
                            max_tokens=min(cfg.resolved_max_tokens, 32768))
        # The canary gets a SHORT budget: its whole point is a fast verdict.
        # A batch queue that cannot answer one tiny request within the probe
        # window is not worth waiting on -- the wave runs realtime instead.
        probe_timeout = float(os.environ.get("AICAP_BATCH_PROBE_TIMEOUT", "300"))
        probe_cfg = replace(batch_cfg, request_timeout=probe_timeout,
                            max_retries=1)
        if label:
            print(f"  batch[{label}]: {len(reqs)} item(s) ({workers} workers) "
                  f"-- probing the Batch Chat endpoint "
                  f"(verdict within ~{int(probe_timeout)}s)")
        first_err: list = []

        def _batch_try(rid, system, user, trycfg):
            """One batch-endpoint attempt; returns (failed, result)."""
            obj, usage, status = self.call_json(system, user, trycfg,
                                                validate=vfn(rid))
            failed = (status == "quarantine" and isinstance(obj, str)
                      and obj.startswith("TRANSPORT_ERROR"))
            if failed and not first_err:
                first_err.append(str(obj)[:300])
            return failed, (obj, usage, status)

        # Canary: ONE item probes the batch endpoint before the wave.  If it
        # fails (or the probe window expires), the whole wave goes straight
        # to the realtime endpoint -- otherwise every item would burn a
        # doomed batch attempt before falling back, and the failure reason
        # would never surface.
        out: dict = {}

        def _finish(rid, res):
            out[rid] = res
            if on_result:
                on_result(rid, res)

        rid0, sys0, usr0 = reqs[0]
        failed0, res0 = _batch_try(rid0, sys0, usr0, probe_cfg)
        if failed0:
            print(f"  batch[{label}]: canary got no batch answer within the "
                  f"probe window -> whole wave via the REALTIME endpoint")
            print(f"    batch probe outcome: "
                  f"{first_err[0] if first_err else 'unknown'}")
            res0 = self.call_json(sys0, usr0, cfg, validate=vfn(rid0))
        else:
            print(f"  batch[{label}]: canary OK -- Batch Chat endpoint live, "
                  f"running the wave at half price")
        _finish(rid0, res0)
        use_batch = not failed0

        fallbacks = []

        def _one(item):
            rid, system, user = item
            if use_batch:
                failed, res = _batch_try(rid, system, user, batch_cfg)
                if failed:
                    fallbacks.append(rid)
                    res = self.call_json(system, user, cfg, validate=vfn(rid))
                return rid, res
            return rid, self.call_json(system, user, cfg, validate=vfn(rid))

        done_n = 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_one, item) for item in reqs[1:]]
            for fut in as_completed(futures):
                rid, res = fut.result()
                _finish(rid, res)
                done_n += 1
                if done_n % 200 == 0 or done_n == len(reqs):
                    print(f"  batch[{label}]: {done_n}/{len(reqs)} done")
        if fallbacks:
            reason = f" (first reason: {first_err[0][:160]})" if first_err else ""
            print(f"  batch[{label}]: {len(fallbacks)} item(s) fell back to "
                  f"the realtime endpoint{reason}")
        return out
