#!/usr/bin/env python3
"""Deterministic time-expression scanner + calendar utilities (the T1 layer).

Rule-first, LLM-for-semantics: this module finds the conventionalized time
expressions of financial text and expands the fully specified ones to
inclusive ISO envelopes.  It never assigns a role or firmness -- that is the
T2 LLM's job.  The T3 validator also uses expand_expression() to re-derive
envelopes for label-type expressions, so calendar arithmetic never depends
on the LLM.

Candidate dict fields (None when not applicable):
  span                exact source substring
  date_start/date_end inclusive ISO envelope when fully resolved
  granularity         day|month|quarter|half_year|year|multi_year|fiscal_year|fiscal_quarter
  bound               exact|end_bound   (end_bound: "by/through/until X" -- start open)
  kind_hint           accounting_period|as_of|None
  fiscal_year         "FYyyyy" label when the expression is fiscal
  fiscal_quarter      1-4 when a fiscal quarter is named
  duration_value/duration_unit  stated tenor ("15-year lease" -> 15, "year")
  needs_anchor        True when a year/date anchor is required to resolve
  pattern             rule name that matched
"""
from __future__ import annotations
import calendar
import datetime as dt
import re

# ---------------------------------------------------------------------------
# Calendar envelope helpers (all inclusive ISO bounds)
# ---------------------------------------------------------------------------


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    return (dt.date(year, month, 1).isoformat(), dt.date(year, month, last).isoformat())


def quarter_bounds(year: int, q: int) -> tuple[str, str]:
    m0 = 3 * (q - 1) + 1
    return (month_bounds(year, m0)[0], month_bounds(year, m0 + 2)[1])


def half_bounds(year: int, h: int) -> tuple[str, str]:
    return ((month_bounds(year, 1)[0], month_bounds(year, 6)[1]) if h == 1
            else (month_bounds(year, 7)[0], month_bounds(year, 12)[1]))


def year_bounds(year: int) -> tuple[str, str]:
    return (dt.date(year, 1, 1).isoformat(), dt.date(year, 12, 31).isoformat())


def quarter_of_date(iso: str) -> tuple[int, int]:
    d = dt.date.fromisoformat(iso)
    return d.year, (d.month - 1) // 3 + 1


def calendar_quarter_of(date_start, date_end):
    """'2026Q3' iff both dates are non-null and inside ONE calendar quarter."""
    if not date_start or not date_end:
        return None
    try:
        y1, q1 = quarter_of_date(date_start)
        y2, q2 = quarter_of_date(date_end)
    except ValueError:
        return None
    return f"{y1}Q{q1}" if (y1, q1) == (y2, q2) else None


def _period_start(end: dt.date, months: int) -> dt.date:
    sm = end.month - months + 1
    sy = end.year + (sm - 1) // 12
    return dt.date(sy, (sm - 1) % 12 + 1, 1)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = (r"(?:January|February|March|April|May|June|July|August|September|"
             r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|"
             r"Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)")
_ORD_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
_YEAR_MIN, _YEAR_MAX = 1990, 2059


def _month_num(tok):
    return _MONTHS.get((tok or "").lower().rstrip("."))


def _year_ok(y):
    """Plausibility bound shared by all loose year captures ("Q3 9999",
    "March 1875" must not become candidates)."""
    return _YEAR_MIN <= y <= _YEAR_MAX


def _fy_year(tok):
    """'2026' -> 2026; \"'26\" / '26' -> 2026 (two-digit fiscal years read as 20xx)."""
    t = str(tok or "").strip().lstrip("'")
    if not t.isdigit():
        return None
    n = int(t)
    if len(t) == 2:
        n += 2000
    return n if _YEAR_MIN <= n <= _YEAR_MAX else None


def _cand(m, **kw):
    base = {"span": m.group(0), "date_start": None, "date_end": None,
            "granularity": None, "bound": "exact", "kind_hint": None,
            "fiscal_year": None, "fiscal_quarter": None, "duration_value": None,
            "duration_unit": None, "needs_anchor": False, "_pos": m.start()}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Pattern builders -- each takes a regex match, returns a candidate or None.
# ---------------------------------------------------------------------------
def _p_period_ended(m):
    n = {"quarter": 3, "three": 3, "3": 3, "six": 6, "6": 6,
         "year": 12, "twelve": 12, "12": 12}.get(m.group(1).lower())
    mon = _month_num(m.group(2))
    if n is None or not mon or not _year_ok(int(m.group(4))):
        return None
    try:
        end = dt.date(int(m.group(4)), mon, int(m.group(3)))
    except ValueError:
        return None
    gran = {3: "quarter", 6: "half_year", 12: "year"}[n]
    return _cand(m, kind_hint="accounting_period", granularity=gran,
                 date_start=_period_start(end, n).isoformat(), date_end=end.isoformat())


def _p_as_of(m):
    sub = scan_time_candidates(m.group(1), _inner=True)
    if len(sub) != 1 or not sub[0]["date_start"]:
        return None
    return _cand(m, kind_hint="as_of", granularity=sub[0]["granularity"],
                 date_start=sub[0]["date_start"], date_end=sub[0]["date_end"])


def _p_by_deadline(m):
    sub = scan_time_candidates(m.group(1), _inner=True)
    if len(sub) != 1 or not sub[0]["date_end"]:
        return None
    return _cand(m, bound="end_bound", granularity=sub[0]["granularity"],
                 date_end=sub[0]["date_end"])          # start deliberately open


def _p_fq_qfy(m):     # "Q3 FY2026" / "Q3 of FY '26"
    fy = _fy_year(m.group(2))
    return None if fy is None else _cand(m, granularity="fiscal_quarter",
                                         fiscal_year=f"FY{fy}", fiscal_quarter=int(m.group(1)),
                                         kind_hint="accounting_period")


def _p_fq_fyq(m):     # "FY2026 Q3"
    fy = _fy_year(m.group(1))
    return None if fy is None else _cand(m, granularity="fiscal_quarter",
                                         fiscal_year=f"FY{fy}", fiscal_quarter=int(m.group(2)),
                                         kind_hint="accounting_period")


def _p_fq_word(m):    # "third quarter of fiscal (year) 2026"
    fy = _fy_year(m.group(2))
    return None if fy is None else _cand(m, granularity="fiscal_quarter",
                                         fiscal_year=f"FY{fy}",
                                         fiscal_quarter=_ORD_Q[m.group(1).lower()],
                                         kind_hint="accounting_period")


def _p_fy(m):
    fy = _fy_year(next(g for g in m.groups() if g))
    return None if fy is None else _cand(m, granularity="fiscal_year",
                                         fiscal_year=f"FY{fy}", kind_hint="accounting_period")


def _p_cq_num(m):     # "Q3 2025" / "3Q 2025"
    q, year = int(m.group(1) or m.group(2)), int(m.group(3))
    if not _year_ok(year):
        return None
    ds, de = quarter_bounds(year, q)
    return _cand(m, granularity="quarter", date_start=ds, date_end=de)


def _p_cq_2d(m):      # "1Q26" / "Q126" / "Q1'26" / "Q1 '26" -> 20xx quarter.
    # Compact analyst shorthand; two-digit years read as 20xx (bounds checked).
    # No FY marker -> CALENDAR quarter candidate, like cq_num; T2 owns role and
    # may still label it fiscal from surrounding evidence.
    q = int(m.group(1) or m.group(2))
    year = _fy_year(m.group(3) or m.group(4))
    if year is None:
        return None
    ds, de = quarter_bounds(year, q)
    return _cand(m, granularity="quarter", date_start=ds, date_end=de)


def _p_cq_word(m):    # "the third quarter of 2025"
    if not _year_ok(int(m.group(2))):
        return None
    ds, de = quarter_bounds(int(m.group(2)), _ORD_Q[m.group(1).lower()])
    return _cand(m, granularity="quarter", date_start=ds, date_end=de)


def _p_half_num(m):   # "H2 2027" / "2H 2027"
    if not _year_ok(int(m.group(3))):
        return None
    ds, de = half_bounds(int(m.group(3)), int(m.group(1) or m.group(2)))
    return _cand(m, granularity="half_year", date_start=ds, date_end=de)


def _p_half_word(m):  # "the second half of 2027"
    if not _year_ok(int(m.group(2))):
        return None
    ds, de = half_bounds(int(m.group(2)), 1 if m.group(1).lower() == "first" else 2)
    return _cand(m, granularity="half_year", date_start=ds, date_end=de)


def _p_mdy(m):
    mon = _month_num(m.group(1))
    if not mon or not _year_ok(int(m.group(3))):
        return None
    try:
        iso = dt.date(int(m.group(3)), mon, int(m.group(2))).isoformat()
    except ValueError:
        return None
    return _cand(m, granularity="day", date_start=iso, date_end=iso)


def _p_iso(m):
    if not _year_ok(int(m.group(1))):
        return None
    try:
        iso = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None
    return _cand(m, granularity="day", date_start=iso, date_end=iso)


def _p_month_year(m):
    mon = _month_num(m.group(1))
    if not mon or not _year_ok(int(m.group(2))):
        return None
    ds, de = month_bounds(int(m.group(2)), mon)
    return _cand(m, granularity="month", date_start=ds, date_end=de)


def _p_year_range(m):
    y1, y2 = int(m.group(1)), int(m.group(2))
    if not (_YEAR_MIN <= y1 <= _YEAR_MAX and y1 < y2 <= _YEAR_MAX and y2 - y1 <= 30):
        return None
    return _cand(m, granularity="multi_year",
                 date_start=year_bounds(y1)[0], date_end=year_bounds(y2)[1])


def _p_duration(m):
    tok = m.group(1).lower()
    n = int(tok) if tok.isdigit() else _WORD_NUM.get(tok)
    if not n or n > 100:
        return None
    return _cand(m, duration_value=n, duration_unit=m.group(2).lower())


def _p_bare_quarter(m):
    return _cand(m, granularity="quarter", needs_anchor=True)


def _p_month_day(m):
    if not _month_num(m.group(1)):
        return None
    return _cand(m, granularity="day", needs_anchor=True)


def _p_bare_year(m):
    year = int(m.group(1))
    ds, de = year_bounds(year)
    return _cand(m, granularity="year", date_start=ds, date_end=de)


# Order matters: most specific first; later patterns never re-consume claimed spans.
_PATTERNS = [
    ("period_ended",
     re.compile(r"\b(quarter|three|3|six|6|year|twelve|12)(?:\s+months?)?\s+"
                r"end(?:ed|ing)\s+(" + _MONTH_RE + r")\s+(\d{1,2}),?\s+(\d{4})", re.I),
     _p_period_ended),
    ("as_of",
     re.compile(r"\bas\s+of\s+((?:" + _MONTH_RE + r")\s+\d{1,2},?\s+\d{4}|"
                r"(?:" + _MONTH_RE + r")\s+\d{4}|\d{4}-\d{2}-\d{2})", re.I),
     _p_as_of),
    ("by_deadline",
     # mid-/early prefixes keep the deadline hint ("by mid-2026" is a
     # deadline, not a bare year); the envelope end stays the coarse period
     # end -- an honest upper bound.
     re.compile(r"\b(?:by|through|until|before)\s+"
                r"(?:the\s+end\s+of\s+|end[- ]|late\s+|mid[- ]?|early\s+)?"
                r"((?:" + _MONTH_RE + r")\s+\d{1,2},?\s+\d{4}|(?:" + _MONTH_RE + r")\s+\d{4}|"
                r"(?:Q[1-4]|[1-4]Q)\s*\d{4}|\d{4})\b", re.I),
     _p_by_deadline),
    ("fq_qfy", re.compile(r"\bQ([1-4])\s*(?:of\s+)?FY\s*('?\d{2,4})\b", re.I), _p_fq_qfy),
    ("fq_fyq", re.compile(r"\bFY\s*('?\d{2,4})\s*Q([1-4])\b", re.I), _p_fq_fyq),
    ("fq_word",
     # two-digit fiscal years accepted ("first quarter of fiscal year '26")
     re.compile(r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\s+of\s+"
                r"fiscal\s+(?:year\s+)?('?\d{2}|\d{4})\b", re.I),
     _p_fq_word),
    ("fiscal_year",
     # "fiscal year '25" / "fiscal year 25" (two-digit after the word "year")
     # match; the (?!\s*%) guard keeps "fiscal year 25%" out.
     re.compile(r"\bFY\s*('?\d{2,4})\b|\bfiscal\s+year\s+('?\d{2}(?!\s*%)|\d{4})\b|"
                r"\bfiscal\s+(\d{4})\b|\bfiscal\s+('\d{2})\b", re.I),
     _p_fy),
    ("cq_num", re.compile(r"\b(?:Q([1-4])|([1-4])Q)\s*(?:of\s+)?(\d{4})\b", re.I), _p_cq_num),
    ("cq_num_2d",
     # compact analyst shorthand -- "1Q26", "Q126", "Q1'26", "Q1 '26".
     # A two-digit year must be attached or apostrophed ("Q1 26" stays
     # unmatched: too ambiguous).
     re.compile(r"\b(?:Q([1-4])|([1-4])Q)(?:\s*'(\d{2})|(\d{2}))\b", re.I),
     _p_cq_2d),
    ("cq_word",
     re.compile(r"\b(first|second|third|fourth)\s+quarter\s+of\s+(\d{4})\b", re.I),
     _p_cq_word),
    ("half_num", re.compile(r"\b(?:H([12])|([12])H)\s*(\d{4})\b", re.I), _p_half_num),
    ("half_word", re.compile(r"\b(first|second)\s+half\s+of\s+(\d{4})\b", re.I), _p_half_word),
    ("month_day_year",
     re.compile(r"\b(" + _MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"),
     _p_mdy),
    ("iso_date", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), _p_iso),
    ("month_year", re.compile(r"\b(" + _MONTH_RE + r")\s+(\d{4})\b"), _p_month_year),
    ("year_range",
     re.compile(r"\b(\d{4})\s*(?:[-–—]|to|through|and)\s*(\d{4})\b", re.I),
     _p_year_range),
    ("duration",
     re.compile(r"\b(\d{1,3}|" + "|".join(_WORD_NUM) + r")[- ](year|month|quarter|week|day)s?\b",
                re.I),
     _p_duration),
    ("bare_quarter",
     re.compile(r"\b(?:Q([1-4])|([1-4])Q)\b(?!\s*(?:of\s+)?(?:FY|'?\d{2,4}))", re.I),
     _p_bare_quarter),
    ("month_day_noyear",
     re.compile(r"\b(" + _MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!,?\s+\d{4})(?!\s*[-–])"),
     _p_month_day),
    ("bare_year", re.compile(r"\b(199\d|20[0-5]\d)\b"), _p_bare_year),
]

_MAX_CANDIDATES = 12


def scan_time_candidates(text, _inner=False):
    """Ordered, non-overlapping candidate scan over one text."""
    if not text or not str(text).strip():
        return []
    text = str(text)
    consumed, out = [], []

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for name, rx, fn in _PATTERNS:
        if _inner and name in ("as_of", "by_deadline"):
            continue                      # wrapper patterns never recurse
        for m in rx.finditer(text):
            rng = (m.start(), m.end())
            if any(overlaps(rng, c) for c in consumed):
                continue
            cand = fn(m)
            if cand is None:
                continue
            cand["pattern"] = name
            consumed.append(rng)
            out.append(cand)
    out.sort(key=lambda c: c["_pos"])
    for c in out:
        c.pop("_pos", None)
    return out[:_MAX_CANDIDATES]


# Tokens that may surround a rule span without changing its envelope.  Any
# other residue -- "now", "late", "early", "mid", "between", "finalized" --
# means the expression carries MORE meaning than the matched pattern, and a
# strict expansion must refuse rather than pretend the span is the whole
# story ("between now and 2028" is NOT the bare year 2028).
_EXPR_FILLER = {"the", "a", "an", "in", "of", "by", "for", "all", "during",
                "within", "on", "at", "to", "and", "s", "as", "from"}


def expand_expression(expr, strict=False):
    """Re-derive the envelope for ONE raw_expression when it is a single fully
    resolved rule pattern covering all of the expression's digits.  With
    strict=True, additionally require every token OUTSIDE the matched span to
    be filler -- the mode the arithmetic tripwire uses, so a span embedded in
    a larger meaning never masquerades as the whole expression.  Loose mode
    remains for fill-only backfills (fiscal labels, stated tenors), which can
    never overwrite a date.  None otherwise."""
    if not expr or " [...] " in str(expr):
        return None
    cands = scan_time_candidates(expr)
    resolved = [c for c in cands
                if c["date_start"] or c["date_end"] or c["fiscal_year"]
                or c["duration_value"]]
    if len(resolved) != 1:
        return None
    c = resolved[0]
    if len(re.findall(r"\d", str(expr))) != len(re.findall(r"\d", c["span"])):
        return None                       # expression carries more than this pattern
    if strict:
        residue = str(expr).replace(c["span"], " ", 1)
        if any(t not in _EXPR_FILLER
               for t in re.split(r"[^\w]+", residue.lower()) if t):
            return None
    return c


# ---------------------------------------------------------------------------
# Deictic expressions -- "this year", "last month", "next quarter", "today",
# "this fiscal year"...  Every pattern here was OBSERVED in the corpus;
# speculative patterns are deliberately absent.  T3 resolves the envelope
# deterministically from doc_date (+ a KNOWN fiscal calendar for the fiscal
# ones), replacing the LLM's largest discretion class (doc_date_relative)
# with code-checked arithmetic.
# ---------------------------------------------------------------------------
_DEICTIC_PATTERNS = [
    # (regex, unit, offset)  -- offset in units relative to the doc_date period
    (re.compile(r"\b(?:earlier|later)?\s*(?:this|the\s+current)\s+year\b", re.I), "year", 0),
    (re.compile(r"\blast\s+year\b", re.I), "year", -1),
    (re.compile(r"\bnext\s+year\b", re.I), "year", 1),
    (re.compile(r"\b(?:this|the\s+current)\s+quarter\b", re.I), "quarter", 0),
    (re.compile(r"\b(?:last|this\s+past|the\s+past)\s+quarter\b", re.I), "quarter", -1),
    (re.compile(r"\bnext\s+quarter\b", re.I), "quarter", 1),
    (re.compile(r"\bthis\s+month\b", re.I), "month", 0),
    (re.compile(r"\blast\s+month\b", re.I), "month", -1),
    (re.compile(r"\bnext\s+month\b", re.I), "month", 1),
    (re.compile(r"\btoday\b", re.I), "day", 0),
    (re.compile(r"\b(?:this|the\s+current)\s+fiscal\s+year\b", re.I), "fiscal_year", 0),
    (re.compile(r"\b(?:last|the\s+prior)\s+fiscal\s+year\b", re.I), "fiscal_year", -1),
]


def scan_deictic(text):
    """Deictic candidates in `text`: [{span, deictic_unit, deictic_offset}]."""
    if not text or not str(text).strip():
        return []
    out, consumed = [], []
    for rx, unit, off in _DEICTIC_PATTERNS:
        for m in rx.finditer(str(text)):
            rng = (m.start(), m.end())
            if any(not (rng[1] <= a or b <= rng[0]) for a, b in consumed):
                continue
            consumed.append(rng)
            out.append({"span": m.group(0), "deictic_unit": unit,
                        "deictic_offset": off, "_pos": m.start()})
    out.sort(key=lambda c: c["_pos"])
    for c in out:
        c.pop("_pos", None)
    return out


# Cheap relative-time cue for the extraction gate: a row whose ONLY time
# evidence is a bare relative phrase in the quote ("last month", "this year")
# must still reach the model -- the scanner above is deliberately blind to
# these (they carry no digits), so the gate needs its own detector.
RELATIVE_CUE_RE = re.compile(
    r"\b(?:this|last|next|earlier|later|coming|past)\s+"
    r"(?:week|month|quarter|year|few\s+years)\b"
    r"|\b(?:today|yesterday|tomorrow)\b", re.I)

# tokens that may surround a deictic phrase without changing its envelope
_DEICTIC_FILLER = {"the", "a", "an", "in", "of", "by", "for", "all", "during",
                   "within", "on", "at", "to", "and", "s"}


def expand_deictic(expr, doc_date, fy_start_month=None):
    """Resolve ONE expression that IS a deictic pattern (modulo filler words)
    against doc_date. Returns {date_start, date_end, granularity} or None.
    Guard: an EMBEDDED deictic ("the second half of this year", "by April
    next year") must NOT be flattened to the bare deictic's envelope -- any
    non-filler residue refuses, leaving the LLM's narrower reading for T4 to
    grade. Fiscal units require a KNOWN calendar (no silent default)."""
    hits = scan_deictic(expr)
    if len(hits) != 1 or not doc_date:
        return None
    residue = str(expr).replace(hits[0]["span"], " ", 1)
    if any(t not in _DEICTIC_FILLER
           for t in re.split(r"[^\w]+", residue.lower()) if t):
        return None
    try:
        d = dt.date.fromisoformat(str(doc_date)[:10])
    except ValueError:
        return None
    unit, off = hits[0]["deictic_unit"], hits[0]["deictic_offset"]
    if unit == "day":
        iso = (d + dt.timedelta(days=off)).isoformat()
        return {"date_start": iso, "date_end": iso, "granularity": "day"}
    if unit == "month":
        m0 = d.month - 1 + off
        y, m = d.year + m0 // 12, m0 % 12 + 1
        ds, de = month_bounds(y, m)
        return {"date_start": ds, "date_end": de, "granularity": "month"}
    if unit == "quarter":
        y, q = quarter_of_date(d.isoformat())
        q0 = (q - 1) + off
        y, q = y + q0 // 4, q0 % 4 + 1
        ds, de = quarter_bounds(y, q)
        return {"date_start": ds, "date_end": de, "granularity": "quarter"}
    if unit == "year":
        ds, de = year_bounds(d.year + off)
        return {"date_start": ds, "date_end": de, "granularity": "year"}
    if unit == "fiscal_year":
        if fy_start_month is None:
            return None                      # unknown calendar: refuse, never guess
        # fiscal year containing doc_date, shifted by `off` years
        fy_start_year = d.year if d.month >= fy_start_month else d.year - 1
        if fy_start_month == 1:
            ds, de = year_bounds(fy_start_year + off)
            return {"date_start": ds, "date_end": de, "granularity": "year"}
        start = dt.date(fy_start_year + off, fy_start_month, 1)
        em = fy_start_month - 1
        ey = start.year + 1
        de = dt.date(ey, em, calendar.monthrange(ey, em)[1])
        return {"date_start": start.isoformat(), "date_end": de.isoformat(),
                "granularity": "fiscal_year"}
    return None


# Reporting-period injection was DELETED in the 2026-08-19 time-layer
# revision: measured on claudetest, 28 of 29 periodic_disclosure rows already
# carried a textual period claim and injection fired once.  Re-add from git
# history only if a full-scale run shows uncovered periodic rows above ~5%.
