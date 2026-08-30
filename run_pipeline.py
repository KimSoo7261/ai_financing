#!/usr/bin/env python3
"""Single entry point for the AICapitalMap pipeline.

Runs extraction (screen -> extract -> audit) and the Fact registry
(intake -> ... -> validate) as one sequenced run.  See README.md for usage.
"""
from aicap.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
