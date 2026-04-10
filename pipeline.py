from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict
from langgraph.graph import END, START, StateGraph

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
    EmbeddingUnavailableError,
    build_index,
    chunk_document,
    classify_source,
    embed_texts,
    load_index,
    retrieve,
    save_index,
)
from schema import (
    CapacityGapCluster,
    ChunkWithMeta,
    CompanyReport,
    ConflictReport,
    EntryPoint,
    FundingCluster,
    GapItem,
    InferenceCluster,
    InferredClaim,
    MessagingParameter,
    OfferingFitCluster,
    OverviewCluster,
    PainItem,
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

logger = logging.getLogger(__name__)


def _merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_lists(left: Optional[list], right: Optional[list]) -> list:
    return list(left or []) + list(right or [])


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
    query_texts: dict[str, str]
    query_vectors: dict[str, list[float]]
    extraction_results: Annotated[dict, _merge_dicts]
    conflicts: Annotated[list[ConflictReport], _merge_lists]
    final_report: Optional[CompanyReport]
    errors: Annotated[list[str], _merge_lists]
    retry_count: int
    output_dir: str
    indices_dir: str
    work_dir: str
    product_context: str


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


def _read_products(product_path: Optional[str | Path] = None) -> list[str]:
    if product_path:
        path = Path(product_path)
        if not path.exists():
            logger.warning("Product profile path does not exist: %s", path)
            return []
        if path.is_file():
            if path.suffix.lower() not in {".md", ".txt", ".json"}:
                logger.warning("Unsupported product profile extension: %s", path)
                return []
            return [path.read_text(encoding="utf-8")]
        products_dir = path
    else:
        products_dir = Path(__file__).resolve().parent / "products"

    if not products_dir.exists():
        return []
    profiles = []
    for path in sorted(products_dir.glob("*.*")):
        if path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        try:
            profiles.append(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load product profile %s: %s", path, exc)
    return profiles


def _format_product_context(profiles: list[str]) -> str:
    return "\n\n---\n\n".join(profiles)


def load_product_context(product_path: Optional[str | Path] = None) -> str:
    return _format_product_context(_read_products(product_path))


def _extract_product_terms(product_context: str, limit: int = 32) -> str:
    if not product_context:
        return ""
    stopwords = {
        "about", "after", "also", "and", "are", "but", "can", "for", "from",
        "have", "into", "not", "our", "that", "the", "their", "this", "with",
        "your", "will", "what", "when", "where", "which", "who", "why",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", product_context.lower())
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        if token not in stopwords:
            counts[token] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return " ".join(token for token, _ in ranked[:limit])


def _base_query_map(product_context: str) -> dict[str, str]:
    product_terms = _extract_product_terms(product_context)
    query_map = {
        "overview": "company overview profile founded headquarters industry business",
        "funding": "company funding raised investors round pe backed",
        "scale": "employees revenue cagr facilities plants geographies footprint",
        "capacity_gaps": (
            f"{product_terms} product relevant capacity gaps bottlenecks manual workflows "
            "disconnected systems operational gaps scale growth hiring expansion visibility "
            "coordination governance compliance customer operations security process maturity"
        ).strip(),
        "pain_points": (
            f"{product_terms} product relevant pain points delays risk compliance "
            "bottlenecks workflow inefficiency scale pressure hiring drag approval bottlenecks "
            "service quality visibility gaps governance readiness expansion complexity"
        ).strip(),
        "triggers": "recent trigger events facility expansion investment leadership product launch",
    }
    return {key: value for key, value in query_map.items() if value}


def _retrieve_from_state(
    state: PipelineState,
    query_key: str,
    fallback_query: str,
    top_k: Optional[int] = None,
    source_type_filter: Optional[str] = None,
) -> list[ChunkWithMeta]:
    query_texts = state.get("query_texts", {})
    query_vectors = state.get("query_vectors", {})
    query = query_texts.get(query_key, fallback_query)
    return retrieve(
        query=query,
        query_vector=query_vectors.get(query_key),
        index=state.get("faiss_index"),
        metadata_store=state.get("faiss_metadata", []),
        top_k=top_k or settings.RETRIEVAL_TOP_K,
        source_type_filter=source_type_filter,
    )


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


def _strategic_summary(extraction_results: dict) -> str:
    summary = {
        section: model.model_dump()
        for section, model in extraction_results.items()
        if hasattr(model, "model_dump") and section != "offering_fit"
    }
    return json.dumps(summary, indent=2)


def _derive_stage(overview: OverviewCluster, funding: FundingCluster, triggers: TriggerCluster) -> Optional[str]:
    if overview.stage:
        return overview.stage

    haystacks: list[str] = []
    if overview.business_description:
        haystacks.append(overview.business_description)
    if funding.last_round_amount:
        haystacks.append(funding.last_round_amount)
    for trigger in triggers.triggers:
        if trigger.event:
            haystacks.append(trigger.event)
        if trigger.significance:
            haystacks.append(trigger.significance)
    text = " ".join(haystacks).lower()

    round_match = re.search(r"\b(series\s+[a-z])\b", text, flags=re.IGNORECASE)
    round_label = round_match.group(1).title() if round_match else None
    is_unicorn = "unicorn" in text
    is_listed = any(token in text for token in ("listed", "ipo", "public"))
    is_pe = "pe-backed" in text or funding.pe_backed is True

    labels: list[str] = []
    if round_label:
        labels.append(round_label)
    if is_pe:
        labels.append("PE-backed")
    if is_listed:
        labels.append("Listed")
    if is_unicorn:
        labels.append("Unicorn")
    if labels:
        return " / ".join(labels)
    return None


def _enrich_overview(
    overview: OverviewCluster,
    funding: FundingCluster,
    scale: ScaleCluster,
    triggers: TriggerCluster,
) -> OverviewCluster:
    if not overview.address and overview.headquarters:
        overview.address = overview.headquarters
    if not overview.employees and scale.employees:
        overview.employees = scale.employees
    if not overview.revenue and scale.revenue:
        overview.revenue = scale.revenue
    if not overview.revenue_cagr and scale.revenue_cagr:
        overview.revenue_cagr = scale.revenue_cagr
    if not overview.stage:
        overview.stage = _derive_stage(overview, funding, triggers)
    return overview


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _dedupe_inferred_items(items: list[GapItem] | list[PainItem]) -> list[GapItem] | list[PainItem]:
    seen: set[str] = set()
    deduped = []
    for item in items:
        key = _normalize_text(item.description)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_strings(values: list[str], limit: Optional[int] = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value.strip())
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _clean_short_phrases(values: list[str], limit: Optional[int] = None) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", value or "").strip(" .:-")
        if not text:
            continue
        if len(text.split()) > 6 or len(text) > 42:
            continue
        cleaned.append(text)
    return _dedupe_strings(cleaned, limit=limit)


def _inference_matches_field(claim: InferredClaim, field_name: str) -> bool:
    normalized = _normalize_text(claim.field_name)
    return normalized == field_name or normalized.startswith(f"{field_name}.")


def _apply_inferences_to_extractions(extraction_results: dict) -> dict:
    updated = dict(extraction_results)
    inferences = updated.get("inferences", InferenceCluster())
    if not getattr(inferences, "claims", None):
        return updated

    capacity_gaps = updated.get("capacity_gaps", CapacityGapCluster())
    pain_points = updated.get("pain_points", PainPointCluster())

    for claim in inferences.claims:
        if _inference_matches_field(claim, "capacity_gaps"):
            capacity_gaps.gaps.append(
                GapItem(
                    description=claim.value,
                    evidence_quote=claim.reasoning,
                    source_type="INFERRED",
                )
            )
            capacity_gaps.source_urls = list(dict.fromkeys([*capacity_gaps.source_urls, *claim.source_urls]))
        elif _inference_matches_field(claim, "pain_points"):
            pain_points.pain_points.append(
                PainItem(
                    description=claim.value,
                    evidence_quote=claim.reasoning,
                    source_type="INFERRED",
                )
            )
            pain_points.source_urls = list(dict.fromkeys([*pain_points.source_urls, *claim.source_urls]))

    capacity_gaps.gaps = _dedupe_inferred_items(capacity_gaps.gaps)[:4]
    pain_points.pain_points = _dedupe_inferred_items(pain_points.pain_points)[:4]

    updated["capacity_gaps"] = capacity_gaps
    updated["pain_points"] = pain_points
    return updated


def _product_motion(product_context: str) -> str:
    lowered = product_context.lower()
    if any(keyword in lowered for keyword in ("interview", "recruit", "hiring", "talent", "assessment", "hr")):
        return "talent"
    if any(keyword in lowered for keyword in ("security", "grc", "compliance", "privacy", "audit", "risk", "infosec", "soc")):
        return "security"
    if any(keyword in lowered for keyword in ("sales", "crm", "lead", "revenue", "marketing", "pipeline")):
        return "revenue"
    if any(keyword in lowered for keyword in ("manufacturing", "operations", "workflow", "plant", "no-code", "automation", "process")):
        return "operations"
    if any(keyword in lowered for keyword in ("finance", "procurement", "spend", "accounts", "invoice")):
        return "finance"
    return "general"


def _motion_title(product_context: str) -> str:
    titles = {
        "talent": "Talent Intelligence",
        "security": "GRC Risk Intelligence",
        "revenue": "Revenue Intelligence",
        "operations": "Operational Intelligence",
        "finance": "Finance Control Intelligence",
        "general": "Strategic Intelligence",
    }
    return titles[_product_motion(product_context)]


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(term in normalized for term in terms)


def _derive_canonical_tags(extraction_results: dict, product_context: str) -> list[str]:
    overview = extraction_results.get("overview", OverviewCluster())
    funding = extraction_results.get("funding", FundingCluster())
    scale = extraction_results.get("scale", ScaleCluster())
    gaps = extraction_results.get("capacity_gaps", CapacityGapCluster()).gaps
    pains = extraction_results.get("pain_points", PainPointCluster()).pain_points
    triggers = extraction_results.get("triggers", TriggerCluster()).triggers
    motion = _product_motion(product_context)

    tags: list[str] = []
    stage = (overview.stage or "").lower()
    if "series" in stage:
        stage_label = overview.stage.strip()
        tags.append(stage_label if len(stage_label) <= 20 else "Growth Stage")
    if "unicorn" in stage:
        tags.append("Unicorn")
    trigger_text = " ".join(f"{item.event} {item.significance or ''}" for item in triggers)
    gap_text = " ".join(item.description for item in gaps)
    pain_text = " ".join(item.description for item in pains)
    all_text = " ".join([trigger_text, gap_text, pain_text, overview.business_description or ""])

    if _contains_any(all_text, ("ipo", "public", "listed")):
        tags.append("IPO-Bound")
    if funding.last_round_amount or funding.total_raised:
        tags.append("Post-Funding")
    if len(scale.geographies) > 1:
        tags.append("Multi-Geo")
    if scale.facilities and _contains_any(scale.facilities, ("plant", "facility", "center")):
        tags.append("Multi-Site")
    if motion == "talent":
        if _contains_any(all_text, ("hiring", "recruit", "talent", "headcount", "interview")):
            tags.append("Hiring at Scale")
        if _contains_any(all_text, ("manual", "round", "screening", "assessment")):
            tags.append("Manual Screening")
        if _contains_any(all_text, ("external", "vendor", "third-party")):
            tags.append("External Tool")
        if _contains_any(all_text, ("ai", "automation")):
            tags.append("AI Readiness")
    elif motion == "security":
        if _contains_any(all_text, ("dpdpa", "hipaa", "iso", "soc 2", "compliance")):
            tags.append("Compliance Pressure")
        if _contains_any(all_text, ("deadline", "may 2027", "audit", "fiduciary")):
            tags.append("Live Deadline")
        if _contains_any(all_text, ("health data", "patient", "privacy", "sensitive")):
            tags.append("Sensitive Data")
        if _contains_any(all_text, ("no framework", "no compliance", "not certified", "external pen")):
            tags.append("Control Gap")
    elif motion == "operations":
        if _contains_any(all_text, ("workflow", "manual", "fragmented", "disconnected")):
            tags.append("Workflow Fragmentation")
        if _contains_any(all_text, ("plant", "facility", "site", "center")):
            tags.append("Plant Complexity")
        if _contains_any(all_text, ("expansion", "jv", "new plant", "new site")):
            tags.append("Expansion Trigger")
        if _contains_any(all_text, ("no-code", "app store", "platform")):
            tags.append("Platform Gap")
    elif motion == "revenue":
        if _contains_any(all_text, ("lead", "conversion", "pipeline", "response speed")):
            tags.append("Pipeline Friction")
        if _contains_any(all_text, ("manual", "sales team", "follow-up")):
            tags.append("Manual Revenue Ops")
        if _contains_any(all_text, ("visibility", "dashboard", "roi")):
            tags.append("Visibility Need")
    elif motion == "finance":
        if _contains_any(all_text, ("approval", "spend", "control", "finance")):
            tags.append("Control Complexity")
        if _contains_any(all_text, ("board", "accountability", "margin", "roi")):
            tags.append("Board Pressure")

    return _dedupe_strings(tags, limit=8)


def _pressure_summary(extraction_results: dict) -> str:
    triggers = extraction_results.get("triggers", TriggerCluster()).triggers
    funding = extraction_results.get("funding", FundingCluster())
    scale = extraction_results.get("scale", ScaleCluster())
    clues: list[str] = []
    if funding.last_round_amount or funding.total_raised:
        clues.append("recent capital and board pressure to convert growth plans into execution")
    if len(scale.geographies) > 1 or scale.facilities:
        clues.append("multi-location or multi-geo operating complexity")
    if scale.employees:
        clues.append("team scale that requires stronger operating infrastructure")
    if triggers:
        clues.append("recent strategic triggers that increase execution urgency")
    return "; ".join(clues) or "current growth and operating complexity"




def _fallback_entry_points(product_context: str, extraction_results: dict) -> list[EntryPoint]:
    motion = _product_motion(product_context)
    pressure = _pressure_summary(extraction_results)
    archetypes: dict[str, list[EntryPoint]] = {
        "talent": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Final approver for strategic infrastructure investments and growth-enablement budget",
                rationale=f"Owns company growth outcomes and must remove execution bottlenecks created by {pressure}",
            ),
            EntryPoint(
                role_title="CHRO/VP of People",
                name=None,
                decision_power="Budget owner for hiring process, talent operations, and recruiting infrastructure",
                rationale="Responsible for scaling hiring systems, improving throughput, and reducing process friction in talent acquisition",
            ),
            EntryPoint(
                role_title="CTO/VP of Engineering",
                name=None,
                decision_power="Approver or influential stakeholder for technical hiring quality, role calibration, and assessment workflow",
                rationale="Needs faster and more reliable capability assessment when growth plans require specialized technical hiring",
            ),
        ],
        "security": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Executive sponsor and final approver for risk, governance, and strategic compliance investments",
                rationale=f"Accountable for reducing enterprise risk and ensuring the business can scale safely under {pressure}",
            ),
            EntryPoint(
                role_title="CISO/CIO/CTO",
                name=None,
                decision_power="Primary technical approver and control owner for security, compliance, and governance tooling",
                rationale="Owns control maturity, audit readiness, and remediation of security or governance gaps",
            ),
            EntryPoint(
                role_title="CFO/COO",
                name=None,
                decision_power="Budget approver for risk reduction, compliance programs, and operating control improvements",
                rationale="Cares about measurable reduction in financial, operational, and regulatory exposure",
            ),
        ],
        "revenue": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Final approver for revenue acceleration and customer-growth investments",
                rationale=f"Needs faster execution and cleaner revenue operations under {pressure}",
            ),
            EntryPoint(
                role_title="CRO/VP of Sales",
                name=None,
                decision_power="Primary budget owner for sales process, pipeline operations, and commercial tooling",
                rationale="Responsible for conversion efficiency, sales productivity, and process consistency across teams",
            ),
            EntryPoint(
                role_title="COO/RevOps Leader",
                name=None,
                decision_power="Operational sponsor for cross-functional process change and systems adoption",
                rationale="Owns execution quality, reporting visibility, and workflow coordination across the go-to-market engine",
            ),
        ],
        "operations": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Final approver for infrastructure and transformation programs that affect operating throughput",
                rationale=f"Needs to remove execution drag and scale operations reliably under {pressure}",
            ),
            EntryPoint(
                role_title="COO/VP of Operations",
                name=None,
                decision_power="Primary budget owner or sponsor for process, plant, workflow, or operations transformation",
                rationale="Owns process efficiency, operational visibility, and reduction of manual coordination across teams or sites",
            ),
            EntryPoint(
                role_title="CTO/CIO/Digital Transformation Leader",
                name=None,
                decision_power="Technical approver for workflow platforms, systems integration, and operational digitization",
                rationale="Responsible for replacing fragmented workflows with scalable digital infrastructure",
            ),
        ],
        "finance": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Final approver for core finance and control-stack investments",
                rationale=f"Needs stronger financial discipline and scalable control processes under {pressure}",
            ),
            EntryPoint(
                role_title="CFO/Finance Director",
                name=None,
                decision_power="Primary budget owner for finance process, control systems, and spend governance",
                rationale="Owns accountability, reporting quality, and operational efficiency across finance workflows",
            ),
            EntryPoint(
                role_title="COO/Procurement Head",
                name=None,
                decision_power="Operational sponsor for cross-functional adoption of approval, vendor, or spend workflows",
                rationale="Cares about process discipline, cycle time, and reduction of fragmented operational decisions",
            ),
        ],
        "general": [
            EntryPoint(
                role_title="CEO/Founder",
                name=None,
                decision_power="Final approver for strategic operating infrastructure investments",
                rationale=f"Must ensure the company can execute growth without bottlenecks under {pressure}",
            ),
            EntryPoint(
                role_title="COO/Business Operations Leader",
                name=None,
                decision_power="Primary operational sponsor for process improvement and execution consistency",
                rationale="Owns cross-functional workflow efficiency, visibility, and adoption of scalable operating systems",
            ),
            EntryPoint(
                role_title="CTO/CIO",
                name=None,
                decision_power="Technical approver for systems, data flow, and platform adoption",
                rationale="Ensures the solution can fit the current stack and improve operational reliability at scale",
            ),
        ],
    }
    return archetypes[motion]


def _enrich_offering_fit(
    offering_fit: OfferingFitCluster,
    extraction_results: dict,
    product_context: str,
) -> OfferingFitCluster:
    canonical_tags = _derive_canonical_tags(extraction_results, product_context)

    if not offering_fit.entry_points:
        offering_fit.entry_points = _fallback_entry_points(product_context, extraction_results)
    else:
        filtered = [
            entry for entry in offering_fit.entry_points
            if any(
                token in entry.decision_power.lower()
                for token in ("approver", "budget", "owner", "sponsor", "sign", "decision", "block")
            )
        ]
        offering_fit.entry_points = (filtered or offering_fit.entry_points)[:3]

    cleaned_signal_tags = _clean_short_phrases(offering_fit.signal_tags, limit=6)
    offering_fit.signal_tags = cleaned_signal_tags or canonical_tags[:6]

    offering_fit.operational_intelligence.tags = _clean_short_phrases(
        offering_fit.operational_intelligence.tags,
        limit=8,
    )

    if not offering_fit.buying_signal_summary and offering_fit.why_now_narrative:
        offering_fit.buying_signal_summary = offering_fit.why_now_narrative

    if not offering_fit.what_we_supply:
        supply_points: list[str] = []
        for match in offering_fit.matched_products:
            if match.fit_rationale:
                supply_points.append(match.fit_rationale)
        for angle in offering_fit.messaging_parameters:
            supply_points.extend(angle.talking_points)
        offering_fit.what_we_supply = _dedupe_strings(supply_points, limit=5)

    offering_fit.additional_intelligence.tags = _clean_short_phrases(
        offering_fit.additional_intelligence.tags,
        limit=6,
    )

    if not offering_fit.recommended_first_move and offering_fit.entry_points:
        primary_entry = offering_fit.entry_points[0]
        first_angle = offering_fit.messaging_parameters[0].angle if offering_fit.messaging_parameters else "the strongest evidence-backed pain"
        offering_fit.recommended_first_move = (
            f"Lead with {first_angle} to the {primary_entry.role_title} and anchor the outreach in the company's clearest execution risk."
        )

    return offering_fit


def node_ingest(state: PipelineState) -> dict:
    logger.info("Running node_ingest")
    chunks: list[ChunkWithMeta] = []
    errors: list[str] = []

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
    indices_dir = state.get("indices_dir") or settings.INDICES_DIR
    cached_index, cached_metadata = load_index(company_id, indices_dir)
    
    if cached_index is not None and cached_metadata is not None:
        return {"faiss_index": cached_index, "faiss_metadata": cached_metadata}

    index, metadata_store = build_index(state.get("chunks", []))
    save_index(index, metadata_store, company_id, indices_dir)
    
    return {"faiss_index": index, "faiss_metadata": metadata_store}


def node_prepare_retrieval_queries(state: PipelineState) -> dict:
    logger.info("Running node_prepare_retrieval_queries")
    product_context = state.get("product_context") or _format_product_context(_read_products())
    query_texts = _base_query_map(product_context)
    if not query_texts:
        return {"query_texts": {}, "query_vectors": {}}

    try:
        vectors = embed_texts(list(query_texts.values()))
    except EmbeddingUnavailableError as exc:
        logger.warning("Query embedding precompute failed: %s", exc)
        return {
            "query_texts": query_texts,
            "query_vectors": {},
            "errors": [f"Embedding query precompute failed: {exc}"],
        }

    query_vectors = {
        name: vector
        for (name, _), vector in zip(query_texts.items(), vectors, strict=False)
    }
    return {"query_texts": query_texts, "query_vectors": query_vectors}


def node_extract_overview(state: PipelineState) -> dict:
    logger.info("Running node_extract_overview")
    context = _retrieve_from_state(
        state,
        query_key="overview",
        fallback_query="company overview profile founded headquarters industry business",
    )
    return {
        "extraction_results": {
            "overview": extract_overview(context, settings.MAX_EXTRACTION_RETRIES)
        }
    }


def node_extract_funding(state: PipelineState) -> dict:
    logger.info("Running node_extract_funding")
    context = _retrieve_from_state(
        state,
        query_key="funding",
        fallback_query="company funding raised investors round pe backed",
    )
    return {
        "extraction_results": {
            "funding": extract_funding(context, settings.MAX_EXTRACTION_RETRIES)
        }
    }


def node_extract_scale(state: PipelineState) -> dict:
    logger.info("Running node_extract_scale")
    context = _retrieve_from_state(
        state,
        query_key="scale",
        fallback_query="employees revenue cagr facilities plants geographies footprint",
    )
    return {
        "extraction_results": {
            "scale": extract_scale(context, settings.MAX_EXTRACTION_RETRIES)
        }
    }


def node_extract_capacity_gaps(state: PipelineState) -> dict:
    logger.info("Running node_extract_capacity_gaps")
    product_context = state.get("product_context") or _format_product_context(_read_products())
    context = _retrieve_from_state(
        state,
        query_key="capacity_gaps",
        fallback_query="product relevant capacity gaps bottlenecks manual workflows disconnected systems operational gaps",
    )
    return {
        "extraction_results": {
            "capacity_gaps": extract_capacity_gaps(
                context,
                product_context=product_context,
                max_retries=settings.MAX_EXTRACTION_RETRIES,
            )
        }
    }


def node_extract_pain_points(state: PipelineState) -> dict:
    logger.info("Running node_extract_pain_points")
    product_context = state.get("product_context") or _format_product_context(_read_products())
    context = _retrieve_from_state(
        state,
        query_key="pain_points",
        fallback_query="product relevant pain points delays risk compliance bottlenecks workflow inefficiency",
    )
    return {
        "extraction_results": {
            "pain_points": extract_pain_points(
                context,
                product_context=product_context,
                max_retries=settings.MAX_EXTRACTION_RETRIES,
            )
        }
    }


def node_extract_triggers(state: PipelineState) -> dict:
    logger.info("Running node_extract_triggers")
    context = _retrieve_from_state(
        state,
        query_key="triggers",
        fallback_query="recent trigger events facility expansion investment leadership product launch",
    )
    return {
        "extraction_results": {
            "triggers": extract_triggers(context, settings.MAX_EXTRACTION_RETRIES)
        }
    }


def node_extract_offering_fit(state: PipelineState) -> dict:
    logger.info("Running node_extract_offering_fit")
    extraction_results = _apply_inferences_to_extractions(state.get("extraction_results", {}))
    product_context = state.get("product_context") or _format_product_context(_read_products())
    gaps = extraction_results.get("capacity_gaps", CapacityGapCluster())
    pains = extraction_results.get("pain_points", PainPointCluster())
    triggers = extraction_results.get("triggers", TriggerCluster())
    inferences = extraction_results.get("inferences")
    inferred_claims = getattr(inferences, "claims", []) if inferences is not None else []
    scale = extraction_results.get("scale", ScaleCluster())
    funding = extraction_results.get("funding", FundingCluster())
    offering_query = " ".join(
        [
            "software vendor solution fit executive leadership decision maker signatory approver budget owner CEO CTO CIO CFO COO CHRO founder managing director VP operations VP digital transformation procurement",
            _extract_product_terms(product_context),
            " ".join(item.description for item in gaps.gaps),
            " ".join(item.description for item in pains.pain_points),
            " ".join(item.event for item in triggers.triggers),
            " ".join(claim.value for claim in inferred_claims),
            scale.employees or "",
            scale.revenue or "",
            scale.facilities or "",
            funding.total_raised or "",
            " ".join(scale.geographies),
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
        strategic_summary=_strategic_summary(extraction_results),
        max_retries=settings.MAX_EXTRACTION_RETRIES,
    )
    extraction_results["offering_fit"] = _enrich_offering_fit(
        extraction_results["offering_fit"],
        extraction_results=extraction_results,
        product_context=product_context,
    )
    return {
        "extraction_results": {
            "capacity_gaps": gaps,
            "pain_points": pains,
            "offering_fit": extraction_results["offering_fit"],
        }
    }


def node_validate_extractions(state: PipelineState) -> dict:
    logger.info("Running node_validate_extractions")
    return {}


def node_infer(state: PipelineState) -> dict:
    logger.info("Running node_infer")
    extraction_results = dict(state.get("extraction_results", {}))
    missing_fields = [
        field_name
        for field_name, value in _collect_field_values(extraction_results).items()
        if value in (None, "", [], {})
    ]
    if not missing_fields:
        return {"extraction_results": extraction_results}
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
    product_context = state.get("product_context") or _format_product_context(_read_products())
    inference_query = " ".join(
        [
            "strategic inference growth hiring expansion governance compliance operations visibility automation",
            _extract_product_terms(product_context),
            " ".join(
                item.event
                for item in extraction_results.get("triggers", TriggerCluster()).triggers
            ),
            extraction_results.get("scale", ScaleCluster()).employees or "",
            extraction_results.get("scale", ScaleCluster()).revenue or "",
            extraction_results.get("funding", FundingCluster()).total_raised or "",
            " ".join(missing_fields),
        ]
    ).strip()
    context = retrieve(
        query=inference_query or "company structural signals inference",
        index=state.get("faiss_index"),
        metadata_store=state.get("faiss_metadata", []),
        top_k=settings.RETRIEVAL_TOP_K,
    )
    extraction_results["inferences"] = extract_inferred_claims(
        context_chunks=context,
        prior_summary=summary,
        product_context=product_context,
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
    overview = _enrich_overview(overview, funding, scale, recent_triggers)

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


def _build_empty_report(state: PipelineState) -> CompanyReport:
    conflicts = state.get("conflicts", [])
    company_domain = state["input"].company_domain
    sections = {
        "overview": _build_report_section("overview", OverviewCluster(), conflicts, company_domain),
        "funding": _build_report_section("funding", FundingCluster(), conflicts, company_domain),
        "scale": _build_report_section("scale", ScaleCluster(), conflicts, company_domain),
        "capacity_gaps": _build_report_section(
            "capacity_gaps", CapacityGapCluster(), conflicts, company_domain
        ),
        "pain_points": _build_report_section(
            "pain_points", PainPointCluster(), conflicts, company_domain
        ),
        "recent_triggers": _build_report_section(
            "recent_triggers", TriggerCluster(), conflicts, company_domain
        ),
        "offering_fit": _build_report_section(
            "offering_fit", OfferingFitCluster(), conflicts, company_domain
        ),
    }
    null_fields_count = sum(_count_nulls(section.data.model_dump()) for section in sections.values())
    return CompanyReport(
        company_id=state["input"].company_id,
        generated_at=_utc_now(),
        overview=sections["overview"],
        funding=sections["funding"],
        scale=sections["scale"],
        capacity_gaps=sections["capacity_gaps"],
        pain_points=sections["pain_points"],
        recent_triggers=sections["recent_triggers"],
        offering_fit=sections["offering_fit"],
        overall_confidence="LOW",
        sources_used=[],
        inferred_claims_count=0,
        null_fields_count=null_fields_count,
        errors=state.get("errors", []),
    )


def node_fail(state: PipelineState) -> dict:
    logger.info("Running node_fail")
    output_dir = _ensure_dir(_output_dir_from_state(state))
    report = _build_empty_report(state)
    output_path = output_dir / f"{state['input'].company_id}.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"final_report": report}


def _post_extract_route(state: PipelineState) -> str:
    extraction_results = state.get("extraction_results", {})
    all_empty = not extraction_results or all(_cluster_is_empty(model) for model in extraction_results.values())
    if len(state.get("errors", [])) > 3 or all_empty:
        return "node_fail"
    return "node_infer"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("node_ingest", node_ingest)
    graph.add_node("node_classify", node_classify)
    graph.add_node("node_chunk", node_chunk)
    graph.add_node("node_embed", node_embed)
    graph.add_node("node_prepare_retrieval_queries", node_prepare_retrieval_queries)
    graph.add_node("node_extract_overview", node_extract_overview)
    graph.add_node("node_extract_funding", node_extract_funding)
    graph.add_node("node_extract_scale", node_extract_scale)
    graph.add_node("node_extract_capacity_gaps", node_extract_capacity_gaps)
    graph.add_node("node_extract_pain_points", node_extract_pain_points)
    graph.add_node("node_extract_triggers", node_extract_triggers)
    graph.add_node("node_validate_extractions", node_validate_extractions)
    graph.add_node("node_infer", node_infer)
    graph.add_node("node_extract_offering_fit", node_extract_offering_fit)
    graph.add_node("node_conflicts", node_conflicts)
    graph.add_node("node_assemble", node_assemble)
    graph.add_node("node_fail", node_fail)

    graph.add_edge(START, "node_ingest")
    graph.add_edge("node_ingest", "node_classify")
    graph.add_edge("node_classify", "node_chunk")
    graph.add_edge("node_chunk", "node_embed")
    graph.add_edge("node_embed", "node_prepare_retrieval_queries")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_overview")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_funding")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_scale")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_capacity_gaps")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_pain_points")
    graph.add_edge("node_prepare_retrieval_queries", "node_extract_triggers")
    graph.add_edge(
        [
            "node_extract_overview",
            "node_extract_funding",
            "node_extract_scale",
            "node_extract_triggers",
            "node_extract_capacity_gaps",
            "node_extract_pain_points",
        ],
        "node_validate_extractions",
    )
    graph.add_conditional_edges(
        "node_validate_extractions",
        _post_extract_route,
        {"node_infer": "node_infer", "node_fail": "node_fail"},
    )
    graph.add_edge("node_infer", "node_extract_offering_fit")
    graph.add_edge("node_extract_offering_fit", "node_conflicts")
    graph.add_edge("node_conflicts", "node_assemble")
    graph.add_edge("node_assemble", END)
    graph.add_edge("node_fail", END)
    return graph.compile()


def run_pipeline(
    pipeline_input: PipelineInput,
    output_dir: Optional[str] = None,
    product_context: str = "",
    product_path: Optional[str | Path] = None,
):
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    resolved_product_context = product_context or _format_product_context(_read_products(product_path))
    work_dir = tempfile.mkdtemp(prefix=f"srbi_{pipeline_input.company_id}_", dir=os.getenv("SRBI_TMP_DIR") or None)
    try:
        state: PipelineState = {
            "input": pipeline_input,
            "chunks": [],
            "faiss_index": None,
            "faiss_metadata": [],
            "query_texts": {},
            "query_vectors": {},
            "extraction_results": {},
            "conflicts": [],
            "final_report": None,
            "errors": [],
            "retry_count": 0,
            "output_dir": output_dir or settings.OUTPUTS_DIR,
            "indices_dir": str(Path(work_dir) / "indices"),
            "work_dir": work_dir,
            "product_context": resolved_product_context,
        }
        graph = build_graph()
        final_state = graph.invoke(state)
        return final_state.get("final_report")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_pipeline_from_documents(
    company_id: str,
    company_domain: str,
    documents: list[dict],
    output_dir: Optional[str] = None,
    product_context: str = "",
    product_path: Optional[str | Path] = None,
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
    return run_pipeline(
        pipeline_input=pipeline_input,
        output_dir=output_dir,
        product_context=product_context,
        product_path=product_path,
    )
