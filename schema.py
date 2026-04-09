from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared base models
# ---------------------------------------------------------------------------


class SRBIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @classmethod
    def empty(cls):
        return cls()


ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
SourceTier = Literal["PRIMARY", "SECONDARY", "INFERRED"]
DirectSourceTier = Literal["PRIMARY", "SECONDARY"]


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class ScrapedFile(SRBIBaseModel):
    url: str
    file_path: str
    scraped_at: str
    raw_text: str


class PipelineInput(SRBIBaseModel):
    company_id: str
    company_domain: str
    files: list[ScrapedFile] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Intermediate models
# ---------------------------------------------------------------------------


class ChunkWithMeta(SRBIBaseModel):
    text: str = ""
    source_url: str = ""
    source_type: DirectSourceTier = "SECONDARY"
    scraped_at: str = ""
    chunk_index: int = 0


class SourcedClaim(SRBIBaseModel):
    value: Optional[str] = None
    source_url: Optional[str] = None
    source_type: SourceTier = "SECONDARY"
    confidence: ConfidenceLevel = "LOW"
    reasoning: Optional[str] = None


class GapItem(SRBIBaseModel):
    description: str = ""
    evidence_quote: Optional[str] = None
    source_type: SourceTier = "INFERRED"


class PainItem(SRBIBaseModel):
    description: str = ""
    evidence_quote: Optional[str] = None
    source_type: SourceTier = "INFERRED"


class TriggerItem(SRBIBaseModel):
    date: Optional[str] = None
    event: str = ""
    significance: Optional[str] = None
    source_type: SourceTier = "INFERRED"


class ProductMatch(SRBIBaseModel):
    product_id: str = ""
    product_name: str = ""
    fit_rationale: str = ""
    evidence_quote: Optional[str] = None
    signal_source_type: SourceTier = "INFERRED"


class EntryPoint(SRBIBaseModel):
    role_title: str = Field(
        description="The executive title, e.g., CTO, CHRO, VP of Operations"
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the executive if found in context",
    )
    decision_power: str = Field(
        description="Why this person can influence or sign the deal, e.g. final approver, budget owner, technical approver, operational sponsor"
    )
    rationale: str = Field(
        description="Why this person cares about the problem (Economic or Primary buyer)"
    )


class MessagingParameter(SRBIBaseModel):
    angle: str = Field(
        description="The core sales angle, e.g., 'IPO Governance Angle', 'Compliance Urgency'"
    )
    talking_points: list[str] = Field(
        default_factory=list,
        description="Specific data points to mention in outreach",
    )


class ConflictReport(SRBIBaseModel):
    field_name: str = ""
    values_found: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    resolution: Literal[
        "HIGHEST_CONFIDENCE_SOURCE", "MOST_RECENT", "UNRESOLVED"
    ] = "UNRESOLVED"


class InferredClaim(SRBIBaseModel):
    field_name: str = ""
    value: str = ""
    source_urls: list[str] = Field(default_factory=list)
    source_type: Literal["INFERRED"] = "INFERRED"
    confidence: ConfidenceLevel = "LOW"
    reasoning: str = ""


class InferenceCluster(SRBIBaseModel):
    claims: list[InferredClaim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction cluster models
# ---------------------------------------------------------------------------


class OverviewCluster(SRBIBaseModel):
    company_name: Optional[str] = None
    founded_year: Optional[str] = None
    headquarters: Optional[str] = None
    industry: Optional[str] = None
    business_description: Optional[str] = None
    source_urls: list[str] = Field(default_factory=list)


class FundingCluster(SRBIBaseModel):
    """
    Summarizes all corporate funding and investment information.
    CRITICAL INSTRUCTION: You MUST return exactly ONE single object. 
    Do NOT return a list or array of multiple funding rounds. Combine all data into this single object.
    """
    total_raised: Optional[str] = Field(None, description="Total money raised across all rounds")
    last_round_amount: Optional[str] = None
    last_round_date: Optional[str] = Field(None, description="Date of the last funding round. MUST be in YYYY-MM-DD format.")
    pe_backed: Optional[bool] = None
    lead_investors: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class ScaleCluster(SRBIBaseModel):
    employees: Optional[str] = None
    revenue: Optional[str] = None
    revenue_cagr: Optional[str] = None
    facilities: Optional[str] = None
    geographies: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class CapacityGapCluster(SRBIBaseModel):
    gaps: list[GapItem] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class PainPointCluster(SRBIBaseModel):
    pain_points: list[PainItem] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class TriggerCluster(SRBIBaseModel):
    triggers: list[TriggerItem] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class OfferingFitCluster(SRBIBaseModel):
    matched_products: list[ProductMatch] = Field(default_factory=list)
    why_now_narrative: Optional[str] = Field(
        default=None,
        description="A 2-3 sentence strategic summary connecting the market trigger to the company's specific pain and our product.",
    )
    entry_points: list[EntryPoint] = Field(default_factory=list)
    messaging_parameters: list[MessagingParameter] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def limit_entry_points(self):
        power_keywords = (
            "chief",
            "ceo",
            "cto",
            "cio",
            "cfo",
            "coo",
            "cdo",
            "cro",
            "cmo",
            "chro",
            "founder",
            "co-founder",
            "managing director",
            "director",
            "president",
            "vice president",
            "vp ",
            "head",
            "gm",
            "general manager",
            "procurement",
            "operations",
            "finance",
            "transformation",
            "technology",
            "digital",
        )
        filtered = [
            entry
            for entry in self.entry_points
            if any(keyword in entry.role_title.lower() for keyword in power_keywords)
        ]
        self.entry_points = (filtered or self.entry_points)[:4]
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class SourceSummary(SRBIBaseModel):
    url: str = ""
    source_type: SourceTier = "SECONDARY"
    used_in_sections: list[str] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


class ReportSection(SRBIBaseModel, Generic[T]):
    data: T
    section_confidence: ConfidenceLevel = "LOW"
    primary_sources_count: int = 0
    secondary_sources_count: int = 0
    inferred_count: int = 0


class CompanyReport(SRBIBaseModel):
    schema_version: str = Field(default="1.0", description="Version of the output schema")
    company_id: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    overview: ReportSection[OverviewCluster]
    funding: ReportSection[FundingCluster]
    scale: ReportSection[ScaleCluster]
    capacity_gaps: ReportSection[CapacityGapCluster]
    pain_points: ReportSection[PainPointCluster]
    recent_triggers: ReportSection[TriggerCluster]
    offering_fit: ReportSection[OfferingFitCluster]
    conflicts: list[ConflictReport] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel = "LOW"
    sources_used: list[SourceSummary] = Field(default_factory=list)
    inferred_claims_count: int = 0
    null_fields_count: int = 0
    errors: list[str] = Field(default_factory=list)
