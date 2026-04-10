# SRBI Extraction Flow
The parent application should call the orchestrator with one company JSON file and one product profile file.

## Main Entrypoint

Use [srbi_orchestrator.py](srbi_orchestrator.py):

```bash
python srbi_orchestrator.py --company-json ./data/leadsquared.json --product-file ./products/interviewgod.md --output-dir ./outputs
```

From another Python module:

```python
from pathlib import Path
from srbi_orchestrator import run_srbi_orchestrator_json

report_json = run_srbi_orchestrator_json(
    company_json=Path("./data/leadsquared.json"),
    product_file=Path("./products/interviewgod.md"),
)

print(report_json["company_id"])
print(report_json["offering_fit"]["data"]["why_now_narrative"])
```

`run_srbi_orchestrator_json(...)` returns a normal Python `dict` that is JSON-serializable. It reads `company_id` and `company_domain` from the JSON file when present, and writes the same payload to a temp output directory by default. Pass `output_dir=Path(...)` only if you want to keep a copy in a specific location. If you need a JSON string instead of a dict, use `run_srbi_orchestrator_json_string(...)`.

If you need the typed Pydantic `CompanyReport` object, use `run_srbi_orchestrator(...)` instead.

The company JSON may be either a list of source records or an object with `company_id`, `company_domain`, and one of `sources`, `files`, `documents`, `records`, or `pages`. Each source record should contain `url`, `scraped_at`, and `text` or `raw_text`.

## Flow

1. `srbi_orchestrator` loads the single company JSON and product `.md`/`.txt` file.
2. `pipeline.run_pipeline` creates a per-run temp directory for FAISS files, suitable for AWS Lambda.
3. LangGraph runs `ingest -> classify -> chunk -> embed`, then micro-batches the base retrieval queries once and reuses those vectors across the parallel overview, funding, scale, capacity-gap, pain-point, and trigger branches.
4. After those base branches finish, the pipeline runs a strategic inference step to derive evidence-backed latent requirements from growth, funding, scale, expansion, and trigger signals.
5. Offering-fit runs after inference so product matching can consider both documented pains and strongly supported latent needs.
6. Product context is injected into capacity-gap, pain-point, inference, and offering-fit extraction, but only as a relevance lens, never as evidence about the company.
7. Product terms are added to FAISS retrieval queries so the retrieved chunks match the product’s ICP and solved problems.
8. Claude is used first for Instructor extraction; Gemini is attempted only as fallback if Claude fails.
9. The final report is written to `{output_dir}/{company_id}.json`.
10. The per-run temp directory is deleted after the pipeline completes.

## Hallucination Guardrails

Product profiles are used as a relevance lens only. Gap and pain extraction must still cite evidence from company context; if company evidence is absent, the extractor should return empty lists rather than inventing product-fit pain.

## Required Environment

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- Optional fallback: `GEMINI_API_KEY`
- Optional tuning: `CLAUDE_EXTRACTION_MODEL`, `GEMINI_EXTRACTION_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_MAX_ITEMS`, `EMBEDDING_BATCH_MAX_TOKENS`, `EMBEDDING_MAX_ATTEMPTS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `MAX_EXTRACTION_RETRIES`, `OUTPUTS_DIR`, `SRBI_TMP_DIR`
