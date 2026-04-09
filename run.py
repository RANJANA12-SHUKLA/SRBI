from __future__ import annotations

import json
from pathlib import Path

import typer

from pipeline import run_pipeline, run_pipeline_from_documents
from schema import PipelineInput, ScrapedFile


app = typer.Typer(add_completion=False, help="Run the SRBI extraction pipeline for one company.")


def _read_json_source(path: Path) -> ScrapedFile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScrapedFile(
        url=payload.get("url", ""),
        file_path=str(path),
        scraped_at=payload.get("scraped_at", ""),
        raw_text=payload.get("text", ""),
    )


def _read_companion_meta(path: Path) -> dict:
    meta_path = path.with_suffix(f"{path.suffix}.meta.json")
    if not meta_path.exists():
        meta_path = path.with_name(f"{path.stem}.meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing companion metadata file for {path.name}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _read_text_or_html_source(path: Path) -> ScrapedFile:
    meta = _read_companion_meta(path)
    return ScrapedFile(
        url=meta.get("url", ""),
        file_path=str(path),
        scraped_at=meta.get("scraped_at", ""),
        raw_text=path.read_text(encoding="utf-8", errors="ignore"),
    )


def load_input_dir(input_dir: Path, company_id: str, company_domain: str) -> PipelineInput:
    files: list[ScrapedFile] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith(".meta.json"):
            continue

        if path.suffix.lower() == ".json":
            files.append(_read_json_source(path))
        elif path.suffix.lower() in {".txt", ".html"}:
            files.append(_read_text_or_html_source(path))

    return PipelineInput(company_id=company_id, company_domain=company_domain, files=files)


def execute_from_directory(
    company_id: str,
    company_domain: str,
    input_dir: Path,
    output_dir: Path | str = Path("./outputs"),
):
    pipeline_input = load_input_dir(input_dir=input_dir, company_id=company_id, company_domain=company_domain)
    return run_pipeline(pipeline_input=pipeline_input, output_dir=str(output_dir))


def execute_from_records(
    company_id: str,
    company_domain: str,
    records: list[dict],
    output_dir: Path | str = Path("./outputs"),
):
    return run_pipeline_from_documents(
        company_id=company_id,
        company_domain=company_domain,
        documents=records,
        output_dir=str(output_dir),
    )


@app.command()
def main(
    company_id: str = typer.Option(..., help="Slug for the target company, e.g. sona-comstar"),
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    company_domain: str = typer.Option(..., help="Company primary domain, e.g. sonacomstar.com"),
    output_dir: Path = typer.Option(Path("./outputs"), help="Directory for final JSON outputs"),
) -> None:
    pipeline_input = load_input_dir(input_dir=input_dir, company_id=company_id, company_domain=company_domain)
    typer.echo(f"[SRBI] Loaded {len(pipeline_input.files)} input files from {input_dir}")

    report = run_pipeline(pipeline_input=pipeline_input, output_dir=str(output_dir))
    if report is None:
        typer.echo("[SRBI] Pipeline completed with partial output only. Check the JSON error log.")
        raise typer.Exit(code=1)

    unresolved_conflicts = len([item for item in report.conflicts if item.resolution == "UNRESOLVED"])
    typer.echo(f"[SRBI] Output written to {output_dir / f'{company_id}.json'}")
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
