"""AICapitalMap pipeline package.

Layout:
    aicap.paths          chunk-set registry, output roots, run tag
    aicap.config         domain taxonomy, fiscal calendars, model registry
    aicap.llm            LLM transport, JSON parsing/salvage, result cache
    aicap.prompts        extraction-stage system prompts and validators
    aicap.classify       deterministic amount parsing and target resolution
    aicap.escalation     audit-evidence routing rules (direction lane)
    aicap.assemble       raw observations -> gated capital-instance rows
    aicap.extraction     Stage 1-3 drivers (screen / extract / audit)
    aicap.time_rules     deterministic time-expression scanner + calendar math
    aicap.time_normalizer  observation-level time claims (gate/read/check/resolve)
    aicap.temporal_profile per-Fact temporal projection (build_profile)
    aicap.contracts      stage names and file contracts
    aicap.orchestrator   stage sequencing for the whole run
    aicap.cli            command-line interface
    aicap.stages         the ten Fact-registry stages plus shared plumbing
"""
