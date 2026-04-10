from __future__ import annotations

import logging
import os
import warnings
from importlib import import_module
from typing import Type, TypeVar

import anthropic
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*google\.generativeai.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    module=r"instructor\.providers\.gemini\.client",
    category=FutureWarning,
)
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

try:
    from utils.secrets_manager import get_secret
except ImportError:  # pragma: no cover - optional in local/dev environments
    get_secret = None


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


BASE_SYSTEM_PROMPT = """You are a structured intelligence extractor for company research reports.

STRICT RULES:
1. You may ONLY use information present in the context chunks below.
2. If a piece of information is not found in the context, return null for that field. Do not guess, infer, or use your training knowledge.
3. For every field you populate, the source_url must come from one of the chunks below.
4. PRODUCT PROFILES are NOT evidence about the company. They are only a relevance lens for deciding whether a company-side signal matters.
5. If you are combining two or more company signals to conclude something no single source explicitly states, you MUST mark it as INFERRED and explain the reasoning. Do not infer from product context alone.
6. If the company evidence is weak, ambiguous, or missing, prefer null or an empty list instead of a speculative answer.
7. Mark confidence HIGH if 2 or more independent sources confirm it. MEDIUM if only one source. LOW if the evidence is ambiguous.
8. Return empty lists, not null, for list fields when no evidence is found.
9. Do not use information from your training data. If the context chunks do not contain this information, return null.

Context chunks (each includes its source URL and source tier):
{context_block}
"""

PROMPTS = {
    "overview": "Extract the company's basic profile. Populate company_name, stage, founded_year, headquarters, address, industry, employees, revenue, revenue_cagr, and business_description when supported by context. Stage should capture the company's maturity label when stated or strongly evidenced, for example Series C, PE-backed, listed, unicorn, pre-IPO, or growth-stage. Business description should be 2-3 sentences summarizing what the company does, who it serves, and its strategic position. Do not copy marketing language; rephrase in neutral analytical language. Do not infer industry, headquarters, founded year, address, stage, or company description from brand familiarity, URL patterns, or product context.",
    "funding": "Extract all investment and funding information. For PE-backed companies, note the PE firm name, investment amount, and any stated mandate or strategic direction the PE board has set. Only capture funding that is explicitly supported by the company context. If dates, amounts, or investor names are unclear, leave them null instead of normalizing from memory.",
    "scale": "Extract scale signals such as employees, revenue, revenue growth, facilities, and geographic footprint. Use only evidence present in the supplied context. Do not estimate or back-solve scale metrics from vague language like 'fast-growing' or 'leading company'.",
    "capacity_gaps": """Identify operational gaps where the company's current infrastructure, process maturity, or tooling may not support its present or near-future scale.
1. EXPLICIT GAPS: Documented bottlenecks such as manual work, fragmented systems, weak visibility, or slow execution.
2. STRATEGIC GAPS: Infer a gap only when company-side signals logically imply a missing capability. Example patterns: rapid hiring can imply recruiting-process strain; multi-geo expansion can imply coordination or visibility gaps; regulatory or audit pressure can imply governance workflow gaps.
3. PRODUCT DISCIPLINE: Use PRODUCT PROFILES only to decide whether the inferred gap is relevant to the product category. Do not infer a gap just because the product can solve it.
Return only generic business problem statements supported by company evidence. Do not invent vendor-specific needs.""",
    "pain_points": """Extract active and latent pain points.
1. DOCUMENTED PAIN: Problems explicitly shown in the context, such as delays, inefficiency, risk, missed visibility, quality issues, hiring drag, poor customer response, or manual process overhead.
2. LATENT PAIN: Risks that logically follow from company trajectory, but only when grounded in company evidence. Example patterns: fundraising can increase accountability and execution pressure; expansion can increase coordination burden; higher complexity can increase reporting, governance, or visibility strain.
3. PRODUCT DISCIPLINE: Use PRODUCT PROFILES only as a relevance lens. Include a pain point only if the company context supports it and the product category plausibly solves it.
For inferred pain points, reason from company-side triggers, scale, or operating complexity. If the evidence does not support the pain, omit it.""",
    "triggers": "Extract recent trigger events that create urgency or change the company's situation. Examples include new facilities, investments, leadership hires, product launches, regulatory filings, and public strategic statements. Include date and significance when available. Do not turn generic company milestones into triggers unless the context shows they are recent and strategically meaningful.",
    "offering_fit": """You have two types of context below: A) COMPANY SIGNALS and B) PRODUCT PROFILES.
Your job is to generate the final strategic sales play.
1. MATCHED PRODUCTS: Recommend ONLY products from the PRODUCT PROFILES that solve the company's documented or strongly supported latent pains/gaps. Do NOT recommend products the company already sells. Do NOT recommend a product unless the company evidence supports the underlying need.
2. WHY NOW NARRATIVE: Write a 2-3 sentence executive summary only when the company context supports a credible buying narrative. You may connect recent triggers, scale, funding, expansion, leadership change, or inferred operational requirements, but only when those are grounded in company evidence. If the evidence is too weak, return null.
3. ENTRY POINTS: Return only the 2-3 strongest deal-driving stakeholders who can realistically sign, approve, budget, sponsor, or block the deal. Prefer roles with clear authority such as CEO, founder, CTO, CIO, CFO, COO, CHRO, business unit head, transformation leader, or compliance/security head depending on the context. If no named executive is present in the context, still return the best-fit role titles with name=null. Do not include weak contacts or generic managers unless the context clearly shows they own the budget or program.
4. OPERATIONAL INTELLIGENCE: Return a compact panel with:
   - tags: 4-8 short tags that summarize the account setup
   - facts: 4-8 labeled facts using concise key-value style entries such as stage, revenue CAGR, plants, geographies, employees, funding, no platform detected, compliance deadline, or expansion count depending on what the evidence supports
5. ADDITIONAL INTELLIGENCE: Return one product-category-aware intelligence box with:
   - title: a short label such as GRC Risk Intelligence, Talent Intelligence, Revenue Intelligence, or Workflow Intelligence
   - tags: 3-6 short tags
   - bullets: 3-5 concise bullets tying the company evidence to the product category
6. WHAT WE SUPPLY: Return 3-5 concise bullets describing what the matched product would supply to this company in practical terms. These should be solution outcomes or capabilities tied directly to the company's evidenced or strongly supported needs.
7. BUYING SIGNAL SUMMARY: Return a short executive summary of why this account is commercially attractive right now.
8. RECOMMENDED FIRST MOVE: Return one concise recommended first move for the sales team, grounded in the strongest evidence and buyer angle.
9. SIGNAL TAGS: Return 3-6 short tags that summarize the opportunity, such as governance pressure, hiring scale, expansion complexity, workflow fragmentation, or security posture.
10. MESSAGING PARAMETERS: Define the sales angle only when it is supported by company context plus the matched product logic. Prefer generic strategic frames such as efficiency, governance, visibility, hiring scale, service quality, security posture, compliance readiness, or expansion readiness depending on the evidence.
11. EVIDENCE DISCIPLINE: PRODUCT PROFILES describe our solution, not the company's reality. Use them only to test fit against company evidence. If the company evidence does not support a fit, return empty matched_products, empty messaging_parameters, empty operational_intelligence facts, empty additional_intelligence bullets, empty what_we_supply, empty signal_tags, and null narratives.
Mark claims INFERRED unless buying intent is explicit.""",
    "infer": """You are generating strategic inferences only. Do not repeat claims already extracted.
Generate only business requirements, latent pains, or operational risks that logically follow from company-side signals.
Good examples of generic inference classes include hiring capacity needs, workflow automation needs, data visibility gaps, governance or compliance pressure, service operations strain, coordination complexity, approval bottlenecks, or security/process maturity gaps.
Only generate a claim if at least two independent company signals together support it, or if one strong trigger clearly implies a near-term operational requirement.
For each claim, set field_name to one of:
- capacity_gaps
- pain_points
- offering_fit
Use capacity_gaps for missing capability or infrastructure needs. Use pain_points for business friction, risk, inefficiency, or pressure. Use offering_fit only for high-level fit logic that does not belong in the other two buckets.
Keep the value generic and company-specific, not product-specific. Never use PRODUCT PROFILES as evidence about the company. Use PRODUCT PROFILES only to prioritize which inferred problem classes matter.
If no logical inference can be supported by company evidence, output an empty list.""",
}

EVIDENCE_CHECKLIST = """Evidence checklist:
- Populate a field only if one or more company context chunks support it.
- If a claim is merely plausible but not evidenced, omit it.
- PRODUCT PROFILES can filter relevance but cannot prove a company fact, pain, gap, trigger, buyer, or fit.
- Prefer an empty list or null over a weak or speculative answer.
- Keep inferred claims conservative and grounded in multiple company-side signals."""

CLAUDE_MODEL = os.getenv("CLAUDE_EXTRACTION_MODEL") or os.getenv(
    "EXTRACTION_MODEL", "claude-sonnet-4-20250514"
)
GEMINI_MODEL = os.getenv("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash")

_claude_client = None
_gemini_client = None


def _load_api_key(secret_name: str) -> str:
    if get_secret is not None:
        try:
            secret_value = get_secret(secret_name)
            if secret_value:
                return str(secret_value)
        except Exception as exc:
            logger.warning("Secret manager lookup failed for %s, falling back to environment: %s", secret_name, exc)
    return os.getenv(secret_name, "")


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        raw_client = anthropic.Anthropic(api_key=_load_api_key("ANTHROPIC_API_KEY"))
        _claude_client = instructor.from_anthropic(raw_client)
    return _claude_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            genai = import_module("google.generativeai")
        except Exception as exc:
            raise RuntimeError("google-generativeai is not installed; Gemini fallback unavailable") from exc
        genai.configure(api_key=_load_api_key("GEMINI_API_KEY"))
        raw_client = genai.GenerativeModel(GEMINI_MODEL)
        _gemini_client = instructor.from_gemini(
            client=raw_client,
            mode=instructor.Mode.GEMINI_JSON,
        )
    return _gemini_client


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
    final_user_prompt = f"{user_prompt}\n\n{EVIDENCE_CHECKLIST}"
    messages = [
        {"role": "system", "content": system_prompt.format(context_block=context_block)},
        {"role": "user", "content": final_user_prompt},
    ]

    try:
        result = _get_claude_client().chat.completions.create(
            model=CLAUDE_MODEL,
            response_model=response_model,
            max_retries=max_retries,
            max_tokens=4096,
            messages=messages,
        )
        return _validate_source_urls(result, context_chunks)
    except Exception as exc:
        logger.warning("Claude extraction failed in %s; trying Gemini fallback: %s", func_name, exc)

    try:
        result = _get_gemini_client().chat.completions.create(
            response_model=response_model,
            max_retries=max_retries,
            messages=messages,
        )
        return _validate_source_urls(result, context_chunks)
    except Exception as exc:
        logger.error("Extraction failed in %s after Claude and Gemini attempts: %s", func_name, exc)
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
    context_chunks: list[ChunkWithMeta], product_context: str = "", max_retries: int = 3
) -> CapacityGapCluster:
    prompt = PROMPTS["capacity_gaps"]
    if product_context:
        prompt = f"{prompt}\n\nPRODUCT PROFILES:\n{product_context}"
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=CapacityGapCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_retries=max_retries,
        func_name="extract_capacity_gaps",
    )


def extract_pain_points(
    context_chunks: list[ChunkWithMeta], product_context: str = "", max_retries: int = 3
) -> PainPointCluster:
    prompt = PROMPTS["pain_points"]
    if product_context:
        prompt = f"{prompt}\n\nPRODUCT PROFILES:\n{product_context}"
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=PainPointCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=prompt,
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
    strategic_summary: str = "",
    max_retries: int = 3,
) -> OfferingFitCluster:
    prompt = PROMPTS["offering_fit"]
    if product_context:
        prompt = f"{prompt}\n\nPRODUCT PROFILES:\n{product_context}"
    if strategic_summary:
        prompt = f"{prompt}\n\nEXTRACTED COMPANY SIGNALS:\n{strategic_summary}"
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
    product_context: str = "",
    max_retries: int = 3,
) -> InferenceCluster:
    prompt = f"{PROMPTS['infer']}\n\nExisting extracted signals:\n{prior_summary}"
    if product_context:
        prompt = f"{prompt}\n\nPRODUCT PROFILES:\n{product_context}"
    return extract_cluster(
        context_chunks=context_chunks,
        response_model=InferenceCluster,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_retries=max_retries,
        func_name="extract_inferred_claims",
    )
