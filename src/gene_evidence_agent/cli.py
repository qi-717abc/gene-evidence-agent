"""Command-line entry point for model-assisted and deterministic evidence queries."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys

from gene_evidence_agent import __version__
from gene_evidence_agent.agent import GeneEvidenceAgent
from gene_evidence_agent.chinese_report import localize_report, TRANSLATION_INPUT_RESERVE_MARGIN
from gene_evidence_agent.config import Settings, get_settings
from gene_evidence_agent.errors import InvalidInputError, ToolError
from gene_evidence_agent.logging_config import configure_logging
from gene_evidence_agent.markdown_report import render_chinese_markdown, render_chinese_status_markdown
from gene_evidence_agent.model_adapter import OpenAICompatibleModelAdapter
from gene_evidence_agent.model_usage import (
    MODEL_LOCALIZATION_TOKEN_BUDGET,
    P5_LOCALIZATION_BUDGET_SCOPE,
)
from gene_evidence_agent.pipeline import GeneEvidencePipeline
from gene_evidence_agent.privacy import ensure_no_sensitive_query
from gene_evidence_agent.schemas import BiomedicalEntity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gene-evidence-agent",
        description="Gene, disease, and drug evidence retrieval agent.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"gene-evidence-agent {__version__}",
    )
    parser.add_argument(
        "--query",
        help="Natural-language request; requires GEA_MODEL_ENABLED=true and the configured model API",
    )
    parser.add_argument("--gene", help="Gene symbol for deterministic, no-language-model retrieval")
    parser.add_argument("--disease", help="Disease name for deterministic, no-language-model retrieval")
    parser.add_argument(
        "--language",
        choices=("en", "zh"),
        default="en",
        help="Output language/view; en preserves the legacy output, zh adds a Chinese evidence view",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format; markdown is an optional human-readable Chinese view",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.format == "markdown" and args.language != "zh":
        parser.error("--format markdown currently requires --language zh")
    if args.query is not None and (args.gene is not None or args.disease is not None):
        parser.error("use --query or the --gene/--disease pair, not both")
    if (args.gene is None) != (args.disease is None):
        parser.error("--gene and --disease must be provided together")
    if args.query is None and args.gene is None:
        parser.print_help()
        return 0

    settings = get_settings()
    try:
        if args.query is not None:
            if not settings.model_enabled:
                raise InvalidInputError(
                    "Model calls are disabled. Set GEA_MODEL_ENABLED=true in the local .env to opt in",
                    source="model",
                )
            ensure_no_sensitive_query(args.query)
            return asyncio.run(
                _run_agent(args.query, settings, language=args.language, output_format=args.format)
            )
        assert args.gene is not None and args.disease is not None
        ensure_no_sensitive_query(f"{args.gene} {args.disease}")
        return asyncio.run(
            _run_deterministic(
                args.gene,
                args.disease,
                settings,
                language=args.language,
                output_format=args.format,
            )
        )
    except (InvalidInputError, ToolError) as error:
        print(f"error[{error.source}/{error.status.value}]: {error}", file=sys.stderr)
        return 2


async def _run_agent(
    query: str,
    settings: Settings,
    *,
    language: str = "en",
    output_format: str = "json",
) -> int:
    adapter = OpenAICompatibleModelAdapter(settings)
    try:
        agent = GeneEvidenceAgent(adapter, settings=settings)
        result = (
            await agent.run(query)
            if language == "en"
            else await agent.run(query, language="zh")
        )
        if output_format == "markdown":
            if result.chinese_report is not None:
                clarification = result.plan.clarification_question if result.plan else None
                print(
                    render_chinese_markdown(
                        result.chinese_report,
                        run_status=result.status,
                        run_warnings=result.warnings,
                        clarification_question=clarification,
                    )
                )
            else:
                clarification = result.plan.clarification_question if result.plan else None
                print(
                    render_chinese_status_markdown(
                        run_status=result.status,
                        run_warnings=result.warnings,
                        clarification_question=clarification,
                    )
                )
        else:
            print(result.model_dump_json(indent=2, exclude_none=True))
        return 1 if result.status == "upstream_error" else 0
    finally:
        await adapter.aclose()


async def _run_deterministic(
    gene_name: str,
    disease_name: str,
    settings: Settings,
    *,
    language: str = "en",
    output_format: str = "json",
) -> int:
    gene = _query_entity("gene", gene_name)
    disease = _query_entity("disease", disease_name)
    collection = await GeneEvidencePipeline(settings).collect(
        gene,
        disease,
        limit=settings.max_results_per_source,
    )
    report = collection.to_evidence_report(query=f"Gene {gene_name}; disease {disease_name}")
    if language == "en":
        print(report.model_dump_json(indent=2, exclude_none=True))
        return 0

    adapter = (
        OpenAICompatibleModelAdapter(
            settings,
            budget_scope=P5_LOCALIZATION_BUDGET_SCOPE,
            input_token_reserve_margin=TRANSLATION_INPUT_RESERVE_MARGIN,
            max_scope_tokens=MODEL_LOCALIZATION_TOKEN_BUDGET,
            max_request_tokens=MODEL_LOCALIZATION_TOKEN_BUDGET,
        )
        if settings.model_enabled
        else None
    )
    try:
        chinese_report = await localize_report(
            report,
            settings=settings,
            adapter=adapter,
            timeout_seconds=settings.agent_run_timeout_seconds,
        )
    finally:
        if adapter is not None:
            await adapter.aclose()
    result = {
        "report": report.model_dump(mode="json", exclude_none=True),
        "chinese_report": chinese_report.model_dump(mode="json", exclude_none=True),
    }
    if output_format == "markdown":
        print(render_chinese_markdown(chinese_report))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _query_entity(entity_type: str, name: str) -> BiomedicalEntity:
    digest = hashlib.sha256(name.casefold().strip().encode("utf-8")).hexdigest()[:20]
    return BiomedicalEntity(
        id=f"cli:{entity_type}:{digest}",
        entity_type=entity_type,  # type: ignore[arg-type]
        query_name=name,
    )


if __name__ == "__main__":
    main()

