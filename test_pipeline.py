"""
Tests for the Critic Agent pipeline.
All LLM calls are mocked -- no API key required.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from schemas import (
    ArtifactType,
    EvaluateRequest,
    RubricDimension,
    RubricOutput,
    CriticOutput,
    DimensionScore,
    GapOutput,
    Gap,
    NextStep,
)
from agents.rubric_builder import get_fallback_rubric
from agents.gap_detector import get_fallback_gap_output
from agents.synthesiser import compute_consistency_score, build_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

GOOD_ARTIFACT = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

SHORT_ARTIFACT = "hello"
GIBBERISH_ARTIFACT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def make_rubric() -> RubricOutput:
    return RubricOutput(
        artifact_type_detected="code",
        dimensions=[
            RubricDimension(name="Correctness", description="Does it work?", weight=2.0),
            RubricDimension(name="Readability", description="Is it clean?", weight=1.0),
            RubricDimension(name="Edge cases", description="Handles edge cases?", weight=1.5),
        ],
    )


def make_critic_output(scores: list[tuple[str, int]] | None = None) -> CriticOutput:
    if scores is None:
        scores = [("Correctness", 4), ("Readability", 5), ("Edge cases", 3)]
    dim_scores = [
        DimensionScore(
            dimension=name, score=score,
            rationale=f"Rationale for {name}",
            evidence=f"Evidence for {name}",
        )
        for name, score in scores
    ]
    weights = {"Correctness": 2.0, "Readability": 1.0, "Edge cases": 1.5}
    weighted = sum(s.score * weights.get(s.dimension, 1.0) for s in dim_scores)
    total_w = sum(weights.get(s.dimension, 1.0) for s in dim_scores)
    return CriticOutput(
        scores=dim_scores,
        overall_score=round(weighted / total_w, 2),
        summary="Decent submission overall.",
    )


def make_gap_output() -> GapOutput:
    return GapOutput(
        gaps=[Gap(area="No tests", description="Missing unit tests", severity="moderate")],
        completeness_score=3,
    )


# ── Validation tests ──────────────────────────────────────────────────────────

class TestRequestValidation:
    def test_valid_request(self):
        req = EvaluateRequest(artifact=GOOD_ARTIFACT, artifact_type=ArtifactType.CODE)
        assert req.artifact == GOOD_ARTIFACT.strip()

    def test_short_artifact_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            EvaluateRequest(artifact=SHORT_ARTIFACT)
        assert "too short" in str(exc_info.value).lower()

    def test_gibberish_artifact_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            EvaluateRequest(artifact=GIBBERISH_ARTIFACT)
        assert "gibberish" in str(exc_info.value).lower()

    def test_artifact_is_stripped(self):
        req = EvaluateRequest(artifact="   " + GOOD_ARTIFACT + "   ")
        assert not req.artifact.startswith(" ")

    def test_auto_type_default(self):
        req = EvaluateRequest(artifact=GOOD_ARTIFACT)
        assert req.artifact_type == ArtifactType.AUTO


# ── Fallback tests ────────────────────────────────────────────────────────────

class TestFallbacks:
    def test_fallback_rubric_has_min_dimensions(self):
        rubric = get_fallback_rubric(ArtifactType.CODE)
        assert len(rubric.dimensions) >= 2

    def test_fallback_gap_output(self):
        gap = get_fallback_gap_output()
        assert len(gap.gaps) >= 1
        assert gap.gaps[0].severity == "minor"

    def test_fallback_rubric_for_each_type(self):
        for t in [ArtifactType.CODE, ArtifactType.BRIEF, ArtifactType.AUTO]:
            rubric = get_fallback_rubric(t)
            assert len(rubric.dimensions) >= 2


# ── Consistency check tests ───────────────────────────────────────────────────

class TestConsistencyCheck:
    def test_identical_outputs_are_reliable(self):
        original = make_critic_output()
        reshuffled = make_critic_output()
        variance, reliable = compute_consistency_score(original, reshuffled)
        assert variance == 0.0
        assert reliable is True

    def test_small_variance_is_reliable(self):
        original = make_critic_output([("Correctness", 4), ("Readability", 5), ("Edge cases", 3)])
        reshuffled = make_critic_output([("Correctness", 4), ("Readability", 4), ("Edge cases", 3)])
        variance, reliable = compute_consistency_score(original, reshuffled)
        assert reliable is True

    def test_large_variance_is_unreliable(self):
        original = make_critic_output([("Correctness", 5), ("Readability", 5), ("Edge cases", 5)])
        reshuffled = make_critic_output([("Correctness", 2), ("Readability", 2), ("Edge cases", 2)])
        variance, reliable = compute_consistency_score(original, reshuffled)
        assert reliable is False
        assert variance > 1.0

    def test_no_shared_dimensions(self):
        original = CriticOutput(
            scores=[DimensionScore(dimension="X", score=4, rationale="r", evidence="e")],
            overall_score=4.0, summary="s"
        )
        reshuffled = CriticOutput(
            scores=[DimensionScore(dimension="Y", score=2, rationale="r", evidence="e")],
            overall_score=2.0, summary="s"
        )
        variance, reliable = compute_consistency_score(original, reshuffled)
        assert reliable is True


# ── Report building tests ─────────────────────────────────────────────────────

class TestBuildReport:
    def test_weighted_score_formula(self):
        rubric = make_rubric()
        critic = make_critic_output()
        gap = make_gap_output()
        next_step = NextStep(
            action="Add edge case tests",
            rationale="Missing tests reduce reliability",
            expected_impact="Edge cases dimension",
        )
        report = build_report(rubric, critic, gap, next_step, "partial")
        expected = round(0.7 * critic.overall_score + 0.3 * gap.completeness_score, 2)
        assert report.weighted_final_score == expected

    def test_report_verdict_propagated(self):
        rubric = make_rubric()
        critic = make_critic_output()
        gap = make_gap_output()
        next_step = NextStep(action="x", rationale="y", expected_impact="z")
        report = build_report(rubric, critic, gap, next_step, "demonstrated")
        assert report.proof_of_work_verdict == "demonstrated"

    def test_consistency_fields(self):
        rubric = make_rubric()
        critic = make_critic_output()
        gap = make_gap_output()
        next_step = NextStep(action="x", rationale="y", expected_impact="z")
        report = build_report(rubric, critic, gap, next_step, "partial", 0.33, True)
        assert report.consistency_score == 0.33
        assert report.consistency_reliable is True


# ── Pipeline integration tests (fully mocked) ────────────────────────────────

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_happy_path(self):
        """Full pipeline with all agents mocked."""
        rubric = make_rubric()
        critic_out = make_critic_output()
        gap_out = make_gap_output()
        next_step = NextStep(action="Add tests", rationale="r", expected_impact="e")

        with (
            patch("app.utils.pipeline.build_rubric", AsyncMock(return_value=rubric)),
            patch("app.utils.pipeline.run_critic", AsyncMock(return_value=critic_out)),
            patch("app.utils.pipeline.detect_gaps", AsyncMock(return_value=gap_out)),
            patch("app.utils.pipeline.synthesise", AsyncMock(return_value=(next_step, "demonstrated"))),
        ):
            from utils.pipeline import run_evaluation_pipeline
            req = EvaluateRequest(artifact=GOOD_ARTIFACT, artifact_type=ArtifactType.CODE)
            report = await run_evaluation_pipeline(req)

        assert report.proof_of_work_verdict == "demonstrated"
        assert report.next_best_step.action == "Add tests"
        assert report.weighted_final_score > 0

    @pytest.mark.asyncio
    async def test_gap_detector_failure_degrades_gracefully(self):
        """Gap detector failure should not crash the pipeline."""
        rubric = make_rubric()
        critic_out = make_critic_output()
        next_step = NextStep(action="Fix edge cases", rationale="r", expected_impact="e")

        with (
            patch("app.utils.pipeline.build_rubric", AsyncMock(return_value=rubric)),
            patch("app.utils.pipeline.run_critic", AsyncMock(return_value=critic_out)),
            patch("app.utils.pipeline.detect_gaps", AsyncMock(side_effect=TimeoutError("timeout"))),
            patch("app.utils.pipeline.synthesise", AsyncMock(return_value=(next_step, "partial"))),
        ):
            from utils.pipeline import run_evaluation_pipeline
            req = EvaluateRequest(artifact=GOOD_ARTIFACT, artifact_type=ArtifactType.CODE)
            report = await run_evaluation_pipeline(req)

        assert report is not None
        assert "unavailable" in report.gaps[0].area.lower()

    @pytest.mark.asyncio
    async def test_critic_failure_raises(self):
        """Critic failure must surface -- we can't evaluate without scores."""
        rubric = make_rubric()

        with (
            patch("app.utils.pipeline.build_rubric", AsyncMock(return_value=rubric)),
            patch("app.utils.pipeline.run_critic", AsyncMock(side_effect=TimeoutError("critic timeout"))),
            patch("app.utils.pipeline.detect_gaps", AsyncMock(return_value=make_gap_output())),
        ):
            from utils.pipeline import run_evaluation_pipeline
            req = EvaluateRequest(artifact=GOOD_ARTIFACT, artifact_type=ArtifactType.CODE)

            with pytest.raises(TimeoutError):
                await run_evaluation_pipeline(req)

    @pytest.mark.asyncio
    async def test_rubric_builder_failure_uses_fallback(self):
        """Rubric builder failure should fall through to fallback rubric."""
        critic_out = make_critic_output()
        gap_out = make_gap_output()
        next_step = NextStep(action="Improve structure", rationale="r", expected_impact="e")

        with (
            patch("app.utils.pipeline.build_rubric", AsyncMock(side_effect=Exception("LLM error"))),
            patch("app.utils.pipeline.run_critic", AsyncMock(return_value=critic_out)),
            patch("app.utils.pipeline.detect_gaps", AsyncMock(return_value=gap_out)),
            patch("app.utils.pipeline.synthesise", AsyncMock(return_value=(next_step, "partial"))),
        ):
            from utils.pipeline import run_evaluation_pipeline
            req = EvaluateRequest(artifact=GOOD_ARTIFACT, artifact_type=ArtifactType.CODE)
            report = await run_evaluation_pipeline(req)

        # Should still succeed with fallback rubric
        assert report is not None
        assert report.proof_of_work_verdict == "partial"