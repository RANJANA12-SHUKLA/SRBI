from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from pipeline import load_product_context, run_pipeline
from schema import PipelineInput, ScrapedFile


app = typer.Typer(
    add_completion=False,
    help="Orchestrate one-company SRBI extraction from a single JSON input file.",
)


def _source_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("Company JSON must be an object or a list of source records.")
    if "url" in payload and any(key in payload for key in ("text", "raw_text", "content", "body")):
        return [payload]
    for key in ("sources", "files", "documents", "records", "pages", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(
        "Company JSON must contain a source list under one of: sources, files, documents, records, pages, items."
    )


def _derive_domain(records: list[dict]) -> str:
    for record in records:
        url = record.get("url") or record.get("source_url")
        if url:
            domain = urlparse(url).netloc.lower().replace("www.", "")
            if domain:
                return domain
    return ""


def _to_scraped_file(record: dict, source_path: Path, index: int) -> ScrapedFile:
    url = record.get("url") or record.get("source_url") or ""
    text = (
        record.get("text")
        or record.get("raw_text")
        or record.get("content")
        or record.get("body")
        or ""
    )
    return ScrapedFile(
        url=url,
        file_path=record.get("file_path") or f"{source_path}#{index}",
        scraped_at=record.get("scraped_at", ""),
        raw_text=text,
    )


def load_company_json(
    company_json: Path,
    company_id: str | None = None,
    company_domain: str | None = None,
) -> PipelineInput:
    payload = json.loads(company_json.read_text(encoding="utf-8-sig"))
    records = _source_records_from_payload(payload)

    resolved_company_id = company_id
    resolved_company_domain = company_domain
    if isinstance(payload, dict):
        resolved_company_id = resolved_company_id or payload.get("company_id") or payload.get("company_name")
        resolved_company_domain = (
            resolved_company_domain
            or payload.get("company_domain")
            or payload.get("domain")
            or payload.get("website_domain")
        )

    resolved_company_id = resolved_company_id or company_json.stem
    resolved_company_domain = resolved_company_domain or _derive_domain(records)
    if not resolved_company_domain:
        raise ValueError("company_domain is required when it cannot be derived from source URLs.")

    return PipelineInput(
        company_id=str(resolved_company_id).strip().lower().replace(" ", "-"),
        company_domain=resolved_company_domain,
        files=[
            _to_scraped_file(record=record, source_path=company_json, index=index)
            for index, record in enumerate(records, start=1)
        ],
    )


def run_srbi_orchestrator(
    company_json: Path,
    product_file: Path | None = None,
    company_id: str | None = None,
    company_domain: str | None = None,
    output_dir: Path | str | None = None,
):
    pipeline_input = load_company_json(
        company_json=company_json,
        company_id=company_id,
        company_domain=company_domain,
    )
    product_context = load_product_context(product_file) if product_file else ""
    resolved_output_dir = output_dir or Path(tempfile.gettempdir()) / "srbi_outputs"
    return run_pipeline(
        pipeline_input=pipeline_input,
        output_dir=str(resolved_output_dir),
        product_context=product_context,
    )


def run_srbi_orchestrator_json(
    company_json: Path,
    product_file: Path | None = None,
    company_id: str | None = None,
    company_domain: str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    pipeline_input = load_company_json(
        company_json=company_json,
        company_id=company_id,
        company_domain=company_domain,
    )
    product_context = load_product_context(product_file) if product_file else ""
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "srbi_outputs"
    report = run_pipeline(
        pipeline_input=pipeline_input,
        output_dir=str(resolved_output_dir),
        product_context=product_context,
    )
    if report is not None:
        return report.model_dump(mode="json")

    output_path = resolved_output_dir / f"{pipeline_input.company_id}.json"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    raise RuntimeError("SRBI pipeline did not return a report and no output JSON was written.")


def run_srbi_orchestrator_json_string(
    company_json: Path,
    product_file: Path | None = None,
    company_id: str | None = None,
    company_domain: str | None = None,
    output_dir: Path | str | None = None,
    indent: int | None = 2,
) -> str:
    report_json = run_srbi_orchestrator_json(
        company_json=company_json,
        product_file=product_file,
        company_id=company_id,
        company_domain=company_domain,
        output_dir=output_dir,
    )
    return json.dumps(report_json, indent=indent, ensure_ascii=False)


@app.command()
def main(
    company_json: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    product_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    company_id: str | None = typer.Option(None, help="Optional company slug override."),
    company_domain: str | None = typer.Option(None, help="Optional primary domain override."),
    output_dir: Path = typer.Option(Path("./outputs"), help="Directory for the final JSON output."),
) -> None:
    report = run_srbi_orchestrator(
        company_json=company_json,
        product_file=product_file,
        company_id=company_id,
        company_domain=company_domain,
        output_dir=output_dir,
    )
    if report is None:
        typer.echo("[SRBI] Pipeline completed with partial output only. Check the JSON error log.")
        raise typer.Exit(code=1)

    unresolved_conflicts = len([item for item in report.conflicts if item.resolution == "UNRESOLVED"])
    typer.echo(f"[SRBI] Output written to {output_dir / f'{report.company_id}.json'}")
    typer.echo(
        "[SRBI] Summary: "
        f"sources={len(report.sources_used)}, "
        f"inferred={report.inferred_claims_count}, "
        f"null_fields={report.null_fields_count}, "
        f"overall_confidence={report.overall_confidence}, "
        f"unresolved_conflicts={unresolved_conflicts}"
    )


if __name__ == "__main__":
    app()
