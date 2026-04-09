from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

import anthropic
import instructor
from pydantic import BaseModel

from schema import (
    CapacityGapCluster,
    ChunkWithMeta,
    FundingCluster,
    InferenceCluster,
    OfferingFitCluster,
    OverviewCluster,
    PainPointCluster,
    ScaleCluster,
    TriggerCluster,
)
from retrieval import format_context_chunks


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


BASE_SYSTEM_PROMPT = """You are a structured intelligence extractor for company research reports.

STRICT RULES:
1. You may ONLY use information present in the context chunks below.
2. If a piece of information is not found in the context, return null for that field. Do not guess, infer, or use your training knowledge.
3. For every field you populate, the source_url must come from one of the chunks below.
4. If you are combining two or more signals to conclude something no single source explicitly states, you MUST set source_type to INFERRED and explain your reasoning in the reasoning field.
5. Mark confidence HIGH if 2 or more independent sources confirm it. MEDIUM if only one source. LOW if the evidence is ambiguous.
6. Return empty lists, not null, for list fields when no evidence is found.
7. Do not use information from your training data. If the context chunks do not contain this information, return null.

Context chunks (each includes its source URL and source tier):
{context_block}
"""

PROMPTS = {
    "overview": "Extract the company's basic profile. Business description should be 2-3 sentences summarizing what the company does, who it serves, and its strategic position. Do not copy marketing language; rephrase in neutral analytical language.",
    "funding": "Extract all investment and funding information. For PE-backed companies, note the PE firm name, investment amount, and any stated mandate or strategic direction the PE board has set. If evidence is absent, leave fields null.",
    "scale": "Extract scale signals such as employees, revenue, revenue growth, facilities, and geographic footprint. Use only evidence present in the supplied context.",
    "capacity_gaps": "Identify specific operational gaps where the company has scale or complexity that its current tools, systems, or IT team cannot serve. Look for IT backlogs, manual workarounds, Excel-based processes, disconnected systems, or leadership comments about transformation challenges. Each gap must have evidence.",
    "pain_points": "Extract documented pain points causing cost, delay, compliance risk, or visibility failures. Look for fragmentation across locations, data silos, reporting delays, compliance issues, and management visibility gaps. Distinguish public evidence from inferred signals.",
    "triggers": "Extract recent trigger events that create urgency or change the company's situation. Examples include new facilities, investments, leadership hires, product launches, regulatory filings, and public strategic statements. Include date and significance when available.",
    "offering_fit": """You have two types of context below: A) COMPANY SIGNALS and B) PRODUCT PROFILES.
Your job is to generate the final strategic sales play.
1. MATCHED PRODUCTS: Recommend ONLY products from the PRODUCT PROFILES that explicitly solve the company's extracted pain points/gaps. Do NOT recommend products the company already sells.
2. WHY NOW NARRATIVE: Write a 2-3 sentence executive summary explaining the "perfect storm" (e.g., combining their recent funding/IPO trigger with their operational pain to explain why they must buy now).
3. ENTRY POINTS: Identify only the 3-4 strongest buying or approval stakeholders who can realistically sign, approve, sponsor, or block the deal. Prefer roles with clear authority such as CEO, CTO, CIO, CFO, COO, CHRO, founder, MD, business unit head, or transformation leader, depending on the context. Do not include weak contacts or generic managers unless the context clearly shows they own the budget or program. For each entry point, explain the decision power they hold and why they care.
4. MESSAGING PARAMETERS: Define the sales angle. Tell the sales rep exactly how to frame the pitch (e.g., 'Lead with the IPO readiness angle, do not use a standard vendor pitch').
Mark claims INFERRED unless buying intent is explicit.""",
    "infer": "You are generating inferences only. Do not repeat claims already extracted. Only generate a claim if at least two independent signals together imply it. If you cannot support an inference, output an empty list.",
}

raw_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = instructor.from_anthropic(raw_client)


def _empty_model(response_model: Type[T]) -> T:
    return response_model()


def _validate_source_urls(result: T, context_chunks: list[ChunkWithMeta]) -> T:
    valid_urls = {chunk.source_url for chunk in context_chunks}
    if not hasattr(result, "source_urls"):
        return result

    source_urls = list(dict.fromkeys(getattr(result, "source_urls", [])))
    invalid = [url for url in source_urls if url not in valid_urls]
    if invalid:
        logger.warning("Filtered hallucinated source URLs: %s", invalid)

    filtered = [url for url in source_urls if url in valid_urls]
    try:
        setattr(result, "source_urls", filtered)
    except Exception:
        return result
    return result


def extract_cluster(
    context_chunks: list[ChunkWithMeta],
    response_model: Type[T],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    func_name: str = "extract_cluster",
) -> T:
    if not context_chunks:
        return _empty_model(response_model)

    context_block = format_context_chunks(context_chunks)

    try:
        result = client.chat.completions.create(
            model=os.getenv("EXTRACTION_MODEL", "claude-3-5-sonnet-20241022"), # Updated to standard Claude 3.5 model string
            response_model=response_model,
            max_retries=max_retries,
            max_tokens=4096, # <--- THIS IS THE FIX
            messages=[
                {"role": "system", "content": system_prompt.format(context_block=context_block)},
                {"role": "user", "content": user_prompt},
            ],
        )

        return _validate_source_urls(result, context_chunks)
    except Exception as exc:
        logger.error("Extraction failed in %s: %s", func_name, exc)
        return _empty_model(response_model)


def extract_overview(context_chunks: list[ChunkWithMeta], max_retries: int = 3) -> OverviewCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=OverviewCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["overview"],
        max_retries=max_retries,
        func_name="extract_overview",
    )


def extract_funding(context_chunks: list[ChunkWithMeta], max_retries: int = 3) -> FundingCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=FundingCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["funding"],
        max_retries=max_retries,
        func_name="extract_funding",
    )


def extract_scale(context_chunks: list[ChunkWithMeta], max_retries: int = 3) -> ScaleCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=ScaleCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["scale"],
        max_retries=max_retries,
        func_name="extract_scale",
    )


def extract_capacity_gaps(
    context_chunks: list[ChunkWithMeta], max_retries: int = 3
) -> CapacityGapCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=CapacityGapCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["capacity_gaps"],
        max_retries=max_retries,
        func_name="extract_capacity_gaps",
    )


def extract_pain_points(
    context_chunks: list[ChunkWithMeta], max_retries: int = 3
) -> PainPointCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=PainPointCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["pain_points"],
        max_retries=max_retries,
        func_name="extract_pain_points",
    )


def extract_triggers(context_chunks: list[ChunkWithMeta], max_retries: int = 3) -> TriggerCluster:
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=TriggerCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=PROMPTS["triggers"],
        max_retries=max_retries,
        func_name="extract_triggers",
    )


def extract_offering_fit(
    context_chunks: list[ChunkWithMeta],
    product_context: str = "",
    max_retries: int = 3,
) -> OfferingFitCluster:
    prompt = PROMPTS["offering_fit"]
    if product_context:
        prompt = f"{prompt}\n\nPRODUCT PROFILES:\n{product_context}"
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=OfferingFitCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_retries=max_retries,
        func_name="extract_offering_fit",
    )


def extract_inferred_claims(
    context_chunks: list[ChunkWithMeta],
    prior_summary: str,
    max_retries: int = 3,
) -> InferenceCluster:
    prompt = f"{PROMPTS['infer']}\n\nExisting extracted signals:\n{prior_summary}"
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=InferenceCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_retries=max_retries,
        func_name="extract_inferred_claims",
    )
