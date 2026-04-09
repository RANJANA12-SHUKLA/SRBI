from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict

from extraction import (
    extract_capacity_gaps,
    extract_funding,
    extract_inferred_claims,
    extract_offering_fit,
    extract_overview,
    extract_pain_points,
    extract_scale,
    extract_triggers,
)
from retrieval import (
    build_index,
    chunk_document,
    classify_source,
    load_index,
    retrieve,
    save_index,
)
from schema import (
    CapacityGapCluster,
    ChunkWithMeta,
    CompanyReport,
    ConflictReport,
    FundingCluster,
    OfferingFitCluster,
    OverviewCluster,
    PainPointCluster,
    PipelineInput,
    ReportSection,
    ScaleCluster,
    SourceSummary,
    TriggerCluster,
)

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency during bootstrap
    BeautifulSoup = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency during bootstrap
    END = "END"
    START = "START"
    StateGraph = None


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    EXTRACTION_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 8
    MAX_EXTRACTION_RETRIES: int = 3
    INDICES_DIR: str = "./indices"
    OUTPUTS_DIR: str = "./outputs"
    LOG_LEVEL: str = "INFO"


settings = Settings()


class PipelineState(TypedDict, total=False):
    input: PipelineInput
    chunks: list[ChunkWithMeta]
    faiss_index: Any
    faiss_metadata: list[dict]
    extraction_results: dict
    conflicts: list[ConflictReport]
    final_report: Optional[CompanyReport]
    errors: list[str]
    retry_count: int
    output_dir: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return "<html" in lowered or "<body" in lowered or "</p>" in lowered or "</div>" in lowered


def _strip_html(text: str) -> str:
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return re.sub(r"<[^>]+>", " ", text).strip()


def _output_dir_from_state(state: PipelineState) -> Path:
    return Path(state.get("output_dir") or settings.OUTPUTS_DIR)


def _ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_products() -> list[str]:
    products_dir = Path("products")
    if not products_dir.exists():
        return []
    profiles = []
    # Reads JSON or MD depending on what you have in the folder
    for path in sorted(products_dir.glob("*.*")):
        if path.suffix not in [".md", ".json"]:
            continue
        try:
            profiles.append(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load product profile %s: %s", path, exc)
    return profiles


def _format_product_context(profiles: list[str]) -> str:
    return "\n\n---\n\n".join(profiles)


def _cluster_is_empty(model) -> bool:
    if model is None:
        return True
    payload = model.model_dump()
    for key, value in payload.items():
        if key == "source_urls":
            continue
        if value not in (None, "", [], {}):
            return False
    return True


def _value_present(value: Any) -> bool:
    return value not in (None, "", [], {}, False)


def _count_nulls(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for item in value.values():
            total += _count_nulls(item)
        return total
    if isinstance(value, list):
        return 0 if value else 1
    return 1 if value is None else 0


def _section_confidence(
    primary_sources_count: int,
    secondary_sources_count: int,
    inferred_count: int,
    unresolved_conflicts: int,
    data_model,
) -> str:
    payload = data_model.model_dump()
    null_count = _count_nulls(payload)
    if primary_sources_count >= 2 and unresolved_conflicts == 0:
        return "HIGH"
    if null_count > max(len(payload), 1) // 2 or inferred_count > (primary_sources_count + secondary_sources_count):
        return "LOW"
    if secondary_sources_count or primary_sources_count:
        return "MEDIUM"
    return "LOW"


def _build_report_section(
    section_name: str,
    data_model,
    conflicts: list[ConflictReport],
    company_domain: str,
):
    source_urls = getattr(data_model, "source_urls", [])
    primary_sources = 0
    secondary_sources = 0
    for source_url in source_urls:
        if source_url.startswith("inferred://"):
            continue
        source_type = classify_source(source_url, company_domain)
        if source_type == "PRIMARY":
            primary_sources += 1
        else:
            secondary_sources += 1

    inferred_count = 0
    payload = data_model.model_dump()
    for value in payload.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    source_type = item.get("source_type") or item.get("signal_source_type")
                    if source_type == "INFERRED":
                        inferred_count += 1

    unresolved_conflicts = len(
        [conflict for conflict in conflicts if conflict.field_name.startswith(section_name)]
    )
    confidence = _section_confidence(
        primary_sources_count=primary_sources,
        secondary_sources_count=secondary_sources,
        inferred_count=inferred_count,
        unresolved_conflicts=unresolved_conflicts,
        data_model=data_model,
    )
    return ReportSection(
        data=data_model,
        section_confidence=confidence,
        primary_sources_count=primary_sources,
        secondary_sources_count=secondary_sources,
        inferred_count=inferred_count,
    )


def _summarize_sources(extraction_results: dict, company_domain: str) -> list[SourceSummary]:
    source_map: dict[str, dict] = defaultdict(lambda: {"source_type": "SECONDARY", "used_in_sections": set()})
    for section_name, model in extraction_results.items():
        if not hasattr(model, "source_urls"):
            continue
        for source_url in getattr(model, "source_urls", []):
            if not source_url:
                continue
            source_map[source_url]["source_type"] = classify_source(source_url, company_domain)
            source_map[source_url]["used_in_sections"].add(section_name)

    return [
        SourceSummary(
            url=url,
            source_type=entry["source_type"],
            used_in_sections=sorted(entry["used_in_sections"]),
        )
        for url, entry in sorted(source_map.items())
    ]


def _collect_field_values(extraction_results: dict) -> dict[str, Any]:
    flattened = {}
    for section_name, model in extraction_results.items():
        if model is None:
            continue
        for key, value in model.model_dump().items():
            if key == "source_urls":
                continue
            flattened[f"{section_name}.{key}"] = value
    return flattened


def _cluster_query_map(state: PipelineState) -> list[tuple[str, str]]:
    return [
        ("overview", "company overview profile founded headquarters industry business"),
        ("funding", "company funding raised investors round pe backed"),
        ("scale", "employees revenue cagr facilities plants geographies footprint"),
        (
            "capacity_gaps",
            "IT backlog manual workflows excel disconnected systems digital transformation operational gaps",
        ),
        (
            "pain_points",
            "operational pain points delays fragmentation HR hiring interview screening bottlenecks"
        ),
        ("triggers", "recent trigger events facility expansion investment leadership product launch"),
    ]


def node_ingest(state: PipelineState) -> dict:
    logger.info("Running node_ingest")
    chunks: list[ChunkWithMeta] = []
    errors = list(state.get("errors", []))

    for scraped_file in state["input"].files:
        text = (scraped_file.raw_text or "").strip()
        if not scraped_file.url:
            errors.append(f"Skipped {scraped_file.file_path}: missing URL")
            continue
        if not text:
            errors.append(f"Skipped {scraped_file.file_path}: empty content")
            continue
        if _looks_like_html(text):
            text = _strip_html(text)
        if not text:
            errors.append(f"Skipped {scraped_file.file_path}: content empty after normalization")
            continue
        chunks.append(
            ChunkWithMeta(
                text=text,
                source_url=scraped_file.url,
                source_type="SECONDARY",
                scraped_at=scraped_file.scraped_at,
                chunk_index=0,
            )
        )

    return {"chunks": chunks, "errors": errors}


def node_classify(state: PipelineState) -> dict:
    logger.info("Running node_classify")
    classified = []
    for chunk in state.get("chunks", []):
        classified.append(
            chunk.model_copy(
                update={
                    "source_type": classify_source(
                        url=chunk.source_url,
                        company_domain=state["input"].company_domain,
                    )
                }
            )
        )
    return {"chunks": classified}


def node_chunk(state: PipelineState) -> dict:
    logger.info("Running node_chunk")
    chunked: list[ChunkWithMeta] = []
    for document in state.get("chunks", []):
        chunked.extend(
            chunk_document(
                text=document.text,
                metadata=document.model_dump(),
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
        )
    return {"chunks": chunked}


def node_embed(state: PipelineState) -> dict:
    logger.info("Running node_embed")
    company_id = state["input"].company_id
    
    # 🚨 FIX: Explicitly passing settings.INDICES_DIR to load_index
    cached_index, cached_metadata = load_index(company_id, settings.INDICES_DIR)
    
    if cached_index is not None and cached_metadata is not None:
        return {"faiss_index": cached_index, "faiss_metadata": cached_metadata}

    index, metadata_store = build_index(state.get("chunks", []))
    
    # 🚨 FIX: Explicitly passing settings.INDICES_DIR to save_index
    save_index(index, metadata_store, company_id, settings.INDICES_DIR)
    
    return {"faiss_index": index, "faiss_metadata": metadata_store}


def node_extract(state: PipelineState) -> dict:
    logger.info("Running node_extract")
    extraction_results = dict(state.get("extraction_results", {}))
    products = _read_products()
    product_context = _format_product_context(products)

    for cluster_name, query in _cluster_query_map(state):
        context = retrieve(
            query=query,
            index=state.get("faiss_index"),
            metadata_store=state.get("faiss_metadata", []),
            top_k=settings.RETRIEVAL_TOP_K,
        )
        if cluster_name == "overview":
            extraction_results[cluster_name] = extract_overview(context, settings.MAX_EXTRACTION_RETRIES)
        elif cluster_name == "funding":
            extraction_results[cluster_name] = extract_funding(context, settings.MAX_EXTRACTION_RETRIES)
        elif cluster_name == "scale":
            extraction_results[cluster_name] = extract_scale(context, settings.MAX_EXTRACTION_RETRIES)
        elif cluster_name == "capacity_gaps":
            extraction_results[cluster_name] = extract_capacity_gaps(
                context, product_context=product_context, max_retries=settings.MAX_EXTRACTION_RETRIES
            )
        elif cluster_name == "pain_points":
            extraction_results[cluster_name] = extract_pain_points(
                context, product_context=product_context, max_retries=settings.MAX_EXTRACTION_RETRIES
            )
        elif cluster_name == "triggers":
            extraction_results[cluster_name] = extract_triggers(context, settings.MAX_EXTRACTION_RETRIES)

    gaps = extraction_results.get("capacity_gaps", CapacityGapCluster())
    pains = extraction_results.get("pain_points", PainPointCluster())
    offering_query = " ".join(
        [
            "software vendor solution fit executive leadership decision maker signatory approver budget owner CEO CTO CIO CFO COO CHRO founder managing director VP operations VP digital transformation procurement",
            " ".join(item.description for item in gaps.gaps),
            " ".join(item.description for item in pains.pain_points),
        ]
    ).strip()
    offering_context = retrieve(
        query=offering_query or "software fit operational gaps pain points",
        index=state.get("faiss_index"),
        metadata_store=state.get("faiss_metadata", []),
        top_k=settings.RETRIEVAL_TOP_K,
    )
    extraction_results["offering_fit"] = extract_offering_fit(
        offering_context,
        product_context=product_context,
        max_retries=settings.MAX_EXTRACTION_RETRIES,
    )

    return {"extraction_results": extraction_results}


def node_infer(state: PipelineState) -> dict:
    logger.info("Running node_infer")
    extraction_results = dict(state.get("extraction_results", {}))
    missing_fields = [
        field_name
        for field_name, value in _collect_field_values(extraction_results).items()
        if value in (None, "", [], {})
    ]
    summary = json.dumps(
        {
            "missing_fields": missing_fields,
            "known_signals": {
                section: model.model_dump()
                for section, model in extraction_results.items()
                if hasattr(model, "model_dump")
            },
        },
        indent=2,
    )
    context = retrieve(
        query=" ".join(["inference", *missing_fields]) or "company structural signals inference",
        index=state.get("faiss_index"),
        metadata_store=state.get("faiss_metadata", []),
        top_k=settings.RETRIEVAL_TOP_K,
    )
    extraction_results["inferences"] = extract_inferred_claims(
        context_chunks=context,
        prior_summary=summary,
        max_retries=settings.MAX_EXTRACTION_RETRIES,
    )
    return {"extraction_results": extraction_results}


def _candidate_values(field_name: str, chunks: list[ChunkWithMeta]) -> list[tuple[str, str, str]]:
    patterns = {
        "scale.employees": r"(\d[\d,\.]*\s*(?:\+)?\s*(?:employees|people|staff))",
        "scale.revenue": r"((?:Rs\.?|INR|\$|USD|₹)\s?[\d,\.]+\s*(?:crore|million|billion)?)",
        "overview.founded_year": r"\b(19\d{2}|20\d{2})\b",
    }
    pattern = patterns.get(field_name)
    if not pattern:
        return []

    found = []
    for chunk in chunks:
        match = re.search(pattern, chunk.text, flags=re.IGNORECASE)
        if match:
            found.append((match.group(1), chunk.source_url, chunk.scraped_at))
    return found


def node_conflicts(state: PipelineState) -> dict:
    logger.info("Running node_conflicts")
    conflicts: list[ConflictReport] = []

    candidate_fields = ["scale.employees", "scale.revenue", "overview.founded_year"]
    for field_name in candidate_fields:
        matches = _candidate_values(field_name, state.get("chunks", []))
        unique_values = list(dict.fromkeys(value for value, _, _ in matches))
        if len(unique_values) <= 1:
            continue

        sources = [source for _, source, _ in matches]
        resolution = "UNRESOLVED"
        if any(classify_source(source, state["input"].company_domain) == "PRIMARY" for source in sources):
            resolution = "HIGHEST_CONFIDENCE_SOURCE"
        elif len({scraped_at for _, _, scraped_at in matches}) > 1:
            resolution = "MOST_RECENT"

        conflicts.append(
            ConflictReport(
                field_name=field_name,
                values_found=unique_values,
                sources=sources,
                resolution=resolution,
            )
        )

    return {"conflicts": conflicts}


def node_assemble(state: PipelineState) -> dict:
    logger.info("Running node_assemble")
    extraction_results = state.get("extraction_results", {})
    conflicts = state.get("conflicts", [])

    overview = extraction_results.get("overview", OverviewCluster())
    funding = extraction_results.get("funding", FundingCluster())
    scale = extraction_results.get("scale", ScaleCluster())
    capacity_gaps = extraction_results.get("capacity_gaps", CapacityGapCluster())
    pain_points = extraction_results.get("pain_points", PainPointCluster())
    recent_triggers = extraction_results.get("triggers", TriggerCluster())
    offering_fit = extraction_results.get("offering_fit", OfferingFitCluster())

    sections = {
        "overview": _build_report_section("overview", overview, conflicts, state["input"].company_domain),
        "funding": _build_report_section("funding", funding, conflicts, state["input"].company_domain),
        "scale": _build_report_section("scale", scale, conflicts, state["input"].company_domain),
        "capacity_gaps": _build_report_section(
            "capacity_gaps", capacity_gaps, conflicts, state["input"].company_domain
        ),
        "pain_points": _build_report_section(
            "pain_points", pain_points, conflicts, state["input"].company_domain
        ),
        "recent_triggers": _build_report_section(
            "recent_triggers", recent_triggers, conflicts, state["input"].company_domain
        ),
        "offering_fit": _build_report_section(
            "offering_fit", offering_fit, conflicts, state["input"].company_domain
        ),
    }
    section_confidences = [section.section_confidence for section in sections.values()]
    confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    overall_confidence = min(section_confidences, key=lambda item: confidence_order[item])

    sources_used = _summarize_sources(extraction_results, state["input"].company_domain)
    inferences = extraction_results.get("inferences")
    inferred_claims_count = len(inferences.claims) if inferences is not None else 0
    null_fields_count = sum(_count_nulls(section.data.model_dump()) for section in sections.values())

    report = CompanyReport(
        company_id=state["input"].company_id,
        generated_at=_utc_now(),
        overview=sections["overview"],
        funding=sections["funding"],
        scale=sections["scale"],
        capacity_gaps=sections["capacity_gaps"],
        pain_points=sections["pain_points"],
        recent_triggers=sections["recent_triggers"],
        offering_fit=sections["offering_fit"],
        conflicts=conflicts,
        overall_confidence=overall_confidence,
        sources_used=sources_used,
        inferred_claims_count=inferred_claims_count,
        null_fields_count=null_fields_count,
        errors=state.get("errors", []),
    )

    output_dir = _ensure_dir(_output_dir_from_state(state))
    output_path = output_dir / f"{state['input'].company_id}.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return {"final_report": report}


def node_fail(state: PipelineState) -> dict:
    logger.info("Running node_fail")
    output_dir = _ensure_dir(_output_dir_from_state(state))
    payload = {
        "company_id": state["input"].company_id,
        "generated_at": _utc_now(),
        "errors": state.get("errors", []),
        "conflicts": [conflict.model_dump() for conflict in state.get("conflicts", [])],
        "partial": True,
    }
    output_path = output_dir / f"{state['input'].company_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"final_report": None}


def _post_extract_route(state: PipelineState) -> str:
    extraction_results = state.get("extraction_results", {})
    all_empty = not extraction_results or all(_cluster_is_empty(model) for model in extraction_results.values())
    if len(state.get("errors", [])) > 3 or all_empty:
        return "node_fail"
    return "node_infer"


class _SequentialGraph:
    def invoke(self, state: PipelineState) -> PipelineState:
        for node in (node_ingest, node_classify, node_chunk, node_embed, node_extract):
            state.update(node(state))
        next_step = _post_extract_route(state)
        if next_step == "node_fail":
            state.update(node_fail(state))
            return state
        for node in (node_infer, node_conflicts, node_assemble):
            state.update(node(state))
        return state


def build_graph():
    if StateGraph is None:
        return _SequentialGraph()

    graph = StateGraph(PipelineState)
    graph.add_node("node_ingest", node_ingest)
    graph.add_node("node_classify", node_classify)
    graph.add_node("node_chunk", node_chunk)
    graph.add_node("node_embed", node_embed)
    graph.add_node("node_extract", node_extract)
    graph.add_node("node_infer", node_infer)
    graph.add_node("node_conflicts", node_conflicts)
    graph.add_node("node_assemble", node_assemble)
    graph.add_node("node_fail", node_fail)

    graph.add_edge(START, "node_ingest")
    graph.add_edge("node_ingest", "node_classify")
    graph.add_edge("node_classify", "node_chunk")
    graph.add_edge("node_chunk", "node_embed")
    graph.add_edge("node_embed", "node_extract")
    graph.add_conditional_edges(
        "node_extract",
        _post_extract_route,
        {"node_infer": "node_infer", "node_fail": "node_fail"},
    )
    graph.add_edge("node_infer", "node_conflicts")
    graph.add_edge("node_conflicts", "node_assemble")
    graph.add_edge("node_assemble", END)
    graph.add_edge("node_fail", END)
    return graph.compile()


def run_pipeline(pipeline_input: PipelineInput, output_dir: Optional[str] = None):
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    state: PipelineState = {
        "input": pipeline_input,
        "chunks": [],
        "faiss_index": None,
        "faiss_metadata": [],
        "extraction_results": {},
        "conflicts": [],
        "final_report": None,
        "errors": [],
        "retry_count": 0,
        "output_dir": output_dir or settings.OUTPUTS_DIR,
    }
    graph = build_graph()
    final_state = graph.invoke(state)
    return final_state.get("final_report")


def run_pipeline_from_documents(
    company_id: str,
    company_domain: str,
    documents: list[dict],
    output_dir: Optional[str] = None,
):
    pipeline_input = PipelineInput(
        company_id=company_id,
        company_domain=company_domain,
        files=[
            {
                "url": document.get("url", ""),
                "file_path": document.get("file_path", ""),
                "scraped_at": document.get("scraped_at", ""),
                "raw_text": document.get("raw_text", document.get("text", "")),
            }
            for document in documents
        ],
    )
    return run_pipeline(pipeline_input=pipeline_input, output_dir=output_dir)