# SRBI Extraction Flow

This folder is not meant to be published as a standalone package. It is a self-contained pipeline module that can sit inside a larger project and be called from application code or from the included CLI.

## What The Parent Project Calls

Use one of these entrypoints:

- `pipeline.run_pipeline(pipeline_input, output_dir=...)` if your project already builds `PipelineInput`
- `pipeline.run_pipeline_from_documents(company_id, company_domain, documents, output_dir=...)` if your project has records in memory
- `run.execute_from_directory(company_id, company_domain, input_dir, output_dir=...)` if your project hands off a folder of scraped files

Each run processes exactly one company and writes:

- `{output_dir}/{company_id}.json`

## End-To-End Flow

1. `node_ingest`: reads raw `.json`, `.txt`, or `.html` source files and normalizes text.
2. `node_classify`: tags every source as `PRIMARY` or `SECONDARY` from URL rules.
3. `node_chunk`: splits each source into overlapping retrieval chunks.
4. `node_embed`: embeds chunks and builds or reloads the FAISS index.
5. `node_extract`: retrieves evidence and runs Instructor extractors for overview, funding, scale, gaps, pain points, triggers, and offering fit.
6. `node_infer`: generates only new `INFERRED` claims from combined signals.
7. `node_conflicts`: flags conflicting values such as revenue or employee-count mismatches.
8. `node_assemble`: computes section confidence, source summaries, and final JSON output.

## Input Expectations

Preferred handoff is one JSON file per source:

```json
{
  "url": "https://techcrunch.com/2026/03/example-company",
  "scraped_at": "2026-03-27",
  "text": "Plain text extracted from the page."
}
```

For `.txt` or `.html`, add a companion `.meta.json` in the same folder with `url` and `scraped_at`.

## Environment And Dependencies

The parent project must provide these env vars before calling the pipeline:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- Optional runtime tuning: `EXTRACTION_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `MAX_EXTRACTION_RETRIES`, `INDICES_DIR`, `OUTPUTS_DIR`

Required libraries are the ones imported by this folder: `anthropic`, `instructor`, `openai`, `langgraph`, `langchain-text-splitters`, `faiss-cpu`, `pydantic`, `pydantic-settings`, `beautifulsoup4`, `tenacity`, `typer`, and `numpy`.

## Notes

- `PRIMARY` means company-controlled or regulatory sources.
- `SECONDARY` means third-party reporting or research databases.
- `INFERRED` means the pipeline combined at least two signals and explains the reasoning.
- If all extractors fail, the pipeline still writes a partial JSON with errors instead of failing silently.
