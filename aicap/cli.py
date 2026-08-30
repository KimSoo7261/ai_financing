"""Command-line interface for the pipeline."""
from __future__ import annotations

import argparse
import json
import os

from . import paths as PC
from .config import model_spec, stage_models
from .contracts import STAGE_CONTRACTS, normalize_stage_name
from .orchestrator import RunOptions, run


def _stage_help() -> str:
    lines = []
    for stage in STAGE_CONTRACTS:
        lines.append(f"{stage.name:28} {stage.description}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import sys
    try:  # live progress even when stdout is redirected into a log file
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="AICapitalMap: extraction -> Fact registry -> quarterly views"
    )
    ap.add_argument("--set", dest="set_id", choices=PC.SETS)
    ap.add_argument("--chunks", dest="chunks_path")
    ap.add_argument(
        "--source-dir",
        help="immutable Stage-3 input directory for downstream-only runs",
    )
    ap.add_argument(
        "--outdir",
        help="run directory; defaults to the current RUN_TAG directory for --set",
    )
    ap.add_argument("--from-stage")
    ap.add_argument("--to-stage")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--views", action="store_true",
                    help="also build the quarterly fact views (held out of "
                         "the default run; the fact table is the product)")
    ap.add_argument("--relations", action="store_true",
                    help="also run the DEFERRED fact-relation stage "
                         "(DEC-027: explicit diagnostic only, real LLM "
                         "spend; never part of the default run)")
    ap.add_argument("--list-stages", action="store_true")
    args = ap.parse_args(argv)

    if args.list_stages:
        print(_stage_help())
        return 0
    if args.screen and args.no_screen:
        ap.error("--screen and --no-screen are mutually exclusive")
    if args.from_stage:
        try:
            normalize_stage_name(args.from_stage)
        except ValueError as exc:
            ap.error(str(exc))
    if args.to_stage:
        try:
            normalize_stage_name(args.to_stage)
        except ValueError as exc:
            ap.error(str(exc))

    outdir = args.outdir
    if not outdir and args.set_id:
        outdir = PC.set_outdir(args.set_id)
    if not outdir:
        ap.error("--outdir is required unless --set is supplied")
    outdir = PC.resolve_outdir(outdir)

    if not args.mock:
        # Every provider used by the configured stage models needs its key.
        required = {model_spec(entry["model_key"]).get("api_key_env")
                    or "ANTHROPIC_API_KEY"
                    for entry in stage_models().values()}
        missing = sorted(v for v in required if not os.environ.get(v))
        if missing:
            ap.error(f"{', '.join(missing)} required for a real run under the "
                     f"configured model profile; use --mock only in scratch")

    screen = True if args.screen else (False if args.no_screen else None)
    # A source snapshot implies a downstream-only run unless the caller names
    # another starting stage explicitly.
    from_stage = args.from_stage or ("intake" if args.source_dir else "screen")
    # DEC-027 release boundary: the default run executes every FACT-LEVEL
    # stage through `validate` (relation/view checks skip when absent); the
    # deferred stages -- build_fact_relations and build_fact_views -- run
    # only when explicitly requested (--relations / --views, or named via
    # --from/--to-stage).  The orchestrator enforces the same rule for direct
    # Python-API callers, so CLI and API defaults are identical.
    report = run(
        RunOptions(
            outdir=outdir,
            chunks_path=args.chunks_path,
            source_dir=args.source_dir,
            set_id=args.set_id,
            from_stage=from_stage,
            to_stage=args.to_stage,
            mock=args.mock,
            batch=not args.no_batch,
            screen=screen,
            limit=args.limit,
            include_news=not args.no_news,
            views=args.views,
            relations=args.relations,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    validation = report.get("validation")
    return 1 if validation is not None and not validation.get("success") else 0
