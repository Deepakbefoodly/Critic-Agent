from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class ArtifactType(str, Enum):
    CODE = "code"
    BRIEF = "brief"
    DRAFT = "draft"
    ESSAY = "essay"
    AUTO = "auto"  # let rubric builder infer


class EvaluateRequest(BaseModel):
    artifact: str = Field(..., min_length=1, description="The learner artifact to evaluate")
    artifact_type: ArtifactType = Field(ArtifactType.AUTO, description="Type of artifact")
    context: Optional[str] = Field(None, description="Optional learning objective or task context")

    @field_validator("artifact")
    @classmethod
    def artifact_must_be_substantial(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 30:
            raise ValueError(
                "Artifact is too short to evaluate meaningfully. "
                "Please submit a more substantive piece of work."
            )
        if len(set(stripped)) < 5:
            raise ValueError(
                "Artifact appears to be gibberish or repetitive content. "
                "Please submit real work for evaluation."
            )
        return stripped


class RubricDimension(BaseModel):
    name: str
    description: str
    weight: float = Field(1.0, ge=0.1, le=3.0)


class RubricOutput(BaseModel):
    artifact_type_detected: str
    dimensions: list[RubricDimension]


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(..., ge=1, le=5)
    rationale: str
    evidence: str  # specific quote/reference from artifact


class CriticOutput(BaseModel):
    scores: list[DimensionScore]
    overall_score: float
    summary: str


class Gap(BaseModel):
    area: str
    description: str
    severity: str  # "critical" | "moderate" | "minor"


class GapOutput(BaseModel):
    gaps: list[Gap]
    completeness_score: int = Field(..., ge=1, le=5)


class NextStep(BaseModel):
    action: str
    rationale: str
    expected_impact: str


class EvaluationReport(BaseModel):
    artifact_type: str
    rubric_dimensions: list[RubricDimension]
    dimension_scores: list[DimensionScore]
    gaps: list[Gap]
    overall_score: float
    completeness_score: int
    weighted_final_score: float
    summary: str
    next_best_step: NextStep
    proof_of_work_verdict: str  # "demonstrated" | "partial" | "not demonstrated"
    consistency_score: Optional[float] = None  # from self-eval
    consistency_reliable: Optional[bool] = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str