import asyncio
import logging

from schemas import EvaluateRequest, EvaluationReport
from agents.rubric_builder import build_rubric, get_fallback_rubric
from agents.critic import run_critic
from agents.gap_detector import detect_gaps, get_fallback_gap_output
from agents.synthesiser import synthesise, compute_consistency_score, build_report

logger = logging.getLogger(__name__)


async def run_evaluation_pipeline(request: EvaluateRequest) -> EvaluationReport:
    """
    Full evaluation pipeline:

    1. Build rubric
    2. Critic + gap detector
    3. Synthesis + self-eval
    4. Assemble final report

    Failure handling:
    - Rubric builder failure  -> fallback rubric, pipeline continues
    - Critic failure          -> re-raises (fatal: can't score without it)
    - Gap detector failure    -> fallback gap output, pipeline continues
    - Synthesis failure       -> re-raises (fatal: can't produce verdict)
    """
    # Step 1: Build Rubric
    logger.info("Building rubric for artifact type: %s", request.artifact_type)
    try:
        rubric = await build_rubric(
            artifact=request.artifact,
            artifact_type=request.artifact_type,
            context=request.context,
        )
        logger.info(
            "Rubric built: %d dimensions, type='%s'",
            len(rubric.dimensions),
            rubric.artifact_type_detected,
        )
    except Exception as e:
        logger.warning("Rubric builder failed (%s). Using fallback rubric.", e)
        rubric = get_fallback_rubric(request.artifact_type)

    # Step 2: Critic + Gap detector in parallel
    logger.info("Running critic and gap detector in parallel")

    results = await asyncio.gather(
        run_critic(
            artifact=request.artifact,
            dimensions=rubric.dimensions,
            artifact_type=rubric.artifact_type_detected,
            context=request.context,
        ),
        detect_gaps(
            artifact=request.artifact,
            artifact_type=rubric.artifact_type_detected,
            context=request.context,
        ),
        return_exceptions=True,
    )

    # Critic is fatal
    if isinstance(results[0], Exception):
        logger.error("Critic agent failed: %s", results[0])
        raise results[0]

    critic_output = results[0]
    logger.info("Critic done. Overall score: %.2f", critic_output.overall_score)

    # Gap detector degrades gracefully
    if isinstance(results[1], Exception):
        logger.warning("Gap detector failed (%s). Using fallback.", results[1])
        gap_output = get_fallback_gap_output()
    else:
        gap_output = results[1]
    logger.info(
        "Gap detector done. %d gaps, completeness: %d/5",
        len(gap_output.gaps),
        gap_output.completeness_score,
    )

    # Step 3: Synthesis + self-eval in parallel
    logger.info("Running synthesis and self-eval in parallel")

    synth_results = await asyncio.gather(
        synthesise(
            rubric=rubric,
            critic_output=critic_output,
            gap_output=gap_output,
            artifact_type=rubric.artifact_type_detected,
        ),
        run_critic(
            artifact=request.artifact,
            dimensions=rubric.dimensions,
            artifact_type=rubric.artifact_type_detected,
            context=request.context,
            shuffle_dimensions=True,
        ),
        return_exceptions=True,
    )

    # Synthesis is fatal
    if isinstance(synth_results[0], Exception):
        logger.error("Synthesis failed: %s", synth_results[0])
        raise synth_results[0]
    next_step, verdict = synth_results[0]

    # Self-eval is optional
    consistency_score = None
    consistency_reliable = None
    if not isinstance(synth_results[1], Exception):
        consistency_score, consistency_reliable = compute_consistency_score(
            critic_output, synth_results[1]
        )
        logger.info(
            "Consistency: mean_var=%.2f, reliable=%s",
            consistency_score, consistency_reliable,
        )
    else:
        logger.warning("Self-eval failed: %s", synth_results[1])

    # Step 4: Final report
    report = build_report(
        rubric=rubric,
        critic_output=critic_output,
        gap_output=gap_output,
        next_step=next_step,
        verdict=verdict,
        consistency_score=consistency_score,
        consistency_reliable=consistency_reliable,
    )
    logger.info(
        "Evaluation complete. verdict=%s, final_score=%.2f",
        report.proof_of_work_verdict,
        report.weighted_final_score,
    )
    return report