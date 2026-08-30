#!/usr/bin/env python3
"""Path configuration: chunk-set registry, output roots, run tag.

RUN_TAG names the output-directory family: a ``--set`` run writes to
``out/out_<RUN_TAG>_<set>``.  Override with the AICAP_RUN_TAG environment
variable when several runs must coexist.

CHUNK_SETS is the registry of named chunk inputs.  ``prescreened`` controls
the Stage-1 default: a pre-screened set seeds every chunk as relevant and
skips the paid relevance screen; the full pool runs the real screen unless
overridden with --screen/--no-screen.
"""
from __future__ import annotations

import os

# ROOT is the pipeline directory (the parent of the aicap package).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIJN_ROOT = os.path.dirname(ROOT)

RUN_TAG = os.environ.get("AICAP_RUN_TAG", "main")

DATA_DIR = os.environ.get("AICAP_DATA_DIR") or ROOT
# 2026-08-20 reorg: the WORKING full pool is chunks_full_v8.1.jsonl in the
# pipeline root (universe v2 corpus, 2021-01..2026-08).  2026-08-28 cleanup:
# pipeline_new keeps only the main pipeline files and the two working input
# corpora; the archived chunk sets live (byte-identical) in
# pipeline_clean/chunks_archive and every named set stays runnable from
# there.  A local chunks_archive/ takes precedence when present.
_LOCAL_ARCHIVE = os.path.join(DATA_DIR, "chunks_archive")
ARCHIVE_DIR = (_LOCAL_ARCHIVE if os.path.isdir(_LOCAL_ARCHIVE)
               else os.path.join(STIJN_ROOT, "pipeline_clean",
                                 "chunks_archive"))
CHUNK_SETS = {
    "sub100": {"file": os.path.join(ARCHIVE_DIR, "chunks_100subsample.jsonl"), "prescreened": True},
    "dirtest10": {"file": os.path.join(ARCHIVE_DIR, "chunks_dirtest10.jsonl"), "prescreened": True},
    "sys150": {"file": os.path.join(ARCHIVE_DIR, "chunks_sys150.jsonl"), "prescreened": False},
    "mrgtest": {"file": os.path.join(ARCHIVE_DIR, "chunks_mrgtest.jsonl"), "prescreened": True},
    "dirfix": {"file": os.path.join(ARCHIVE_DIR, "chunks_dirfix.jsonl"), "prescreened": True},
    "claudetest": {"file": os.path.join(ARCHIVE_DIR, "chunks_claudetest.jsonl"), "prescreened": True},
    "richdup": {"file": os.path.join(ARCHIVE_DIR, "chunks.jsonl"), "prescreened": True},
    # 2026-08-29 gold-standard evaluation sample: 100 random screen-relevant
    # 2025 chunks; annotations in ../gold_standard_2025/gold/ (see its README).
    "gold100": {"file": os.path.join(STIJN_ROOT, "gold_standard_2025", "sample",
                                     "chunks_gold100.jsonl"),
                "prescreened": True},
    "full": {"file": "chunks_full_v8.1.jsonl", "prescreened": False},
    "full_v80": {"file": os.path.join(STIJN_ROOT, "data", "chunks", "full_pool_v8.0",
                                      "chunks.jsonl"),
                 "prescreened": False},
}
SETS = tuple(CHUNK_SETS)

OUT_ROOT = os.environ.get("AICAP_OUT_ROOT") or os.path.join(ROOT, "out")


def chunks_path(setid: str) -> str:
    """Absolute path to a named chunk set."""
    try:
        f = CHUNK_SETS[setid]["file"]
    except KeyError:
        raise KeyError(f"unknown chunk set {setid!r}; known: {sorted(CHUNK_SETS)}")
    return f if os.path.isabs(f) else os.path.join(DATA_DIR, f)


def is_prescreened(setid: str) -> bool:
    return bool(CHUNK_SETS.get(setid, {}).get("prescreened"))


def set_outdir(setid: str) -> str:
    """The output directory for a named set: out/out_<RUN_TAG>_<set>."""
    return os.path.join(OUT_ROOT, f"out_{RUN_TAG}_{setid}")


def resolve_outdir(outdir: str) -> str:
    """An absolute path is used as-is; a bare name lands under OUT_ROOT."""
    if os.path.isabs(outdir):
        return outdir
    return os.path.join(OUT_ROOT, outdir)


if __name__ == "__main__":
    print("ROOT     =", ROOT)
    print("RUN_TAG  =", RUN_TAG)
    for setid in SETS:
        print(f"chunks[{setid}] =", chunks_path(setid),
              "| prescreened =", is_prescreened(setid),
              "| outdir =", set_outdir(setid))
    print("OUT_ROOT =", OUT_ROOT)
