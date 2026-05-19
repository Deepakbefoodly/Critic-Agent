import json
import asyncio
import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import (
    CriticOutput, GapOutput, RubricOutput,
    NextStep, EvaluationReport,
)
from utils.llm_client import get_critic_llm
from config import get_settings

SYNTHESIS_SYSTEM_PROMPT = """You are a learning coach synthesising an evaluation report for a learner.

You have received dimension scores from the Critic agent and gap analysis from the Gap Detector agent.

Your job is to:
- Identify the single most impactful "next best step" for the learner
- Determine the proof-of-work verdict: did this learner demonstrate competency?

Proof-of-work verdicts:
  "demonstrated"     -- overall_score >= 3.5 AND completeness_score >= 3 AND no critical gaps
  "partial"          -- overall_score 2.5-3.4 OR has 1 critical gap but real effort is evident
  "not demonstrated" -- overall_score < 2.5 OR 2+ critical gaps OR clearly insufficient work

Next step must be:
- Specific and actionable (not "improve your code")
- Achievable in one focused session
- The highest-leverage improvement for THIS artifact

Respond ONLY with valid JSON. No markdown, no explanation.

JSON schema:
{
  "next_best_step": {
    "action": "string (what to do, specifically)",
    "rationale": "string (why this is the highest-leverage improvement)",
    "expected_impact": "string (what score or dimension it would most improve)"
  },
  "proof_of_work_verdict": "demonstrated|partial|not demonstrated"
}
"""


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Synthesis returned non-JSON: {raw[:200]}")


async def synthesise(
    rubric: RubricOutput,
    critic_output: CriticOutput,
    gap_output: GapOutput,
    artifact_type: str,
) -> tuple[NextStep, str]:
    """
    Merge critic scores + gap analysis -> next step + proof-of-work verdict.
    """
    settings = get_settings()
    llm = get_critic_llm()

    scores_text = "\n".join(
        f"- {s.dimension}: {s.score}/5 -- {s.rationale}" for s in critic_output.scores
    )
    gaps_text = "\n".join(
        f"- [{g.severity.upper()}] {g.area}: {g.description}" for g in gap_output.gaps
    )

    user_content = (
        f"Artifact type: {artifact_type}\n"
        f"Overall score (weighted): {critic_output.overall_score}/5\n"
        f"Completeness score: {gap_output.completeness_score}/5\n\n"
        f"Dimension scores:\n{scores_text}\n\n"
        f"Identified gaps:\n{gaps_text}\n\n"
        f"Critic summary: {critic_output.summary}"
    )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=settings.agent_timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Synthesis agent timed out.")

    raw = response.content.strip()
    data = _parse_json(raw)

    next_step = NextStep(**data["next_best_step"])
    verdict = data.get("proof_of_work_verdict", "partial")
    return next_step, verdict


def compute_consistency_score(
    original: CriticOutput, reshuffled: CriticOutput
) -> tuple[float, bool]:
    orig_map = {s.dimension: s.score for s in original.scores}
    new_map = {s.dimension: s.score for s in reshuffled.scores}
    shared = set(orig_map.keys()) & set(new_map.keys())
    if not shared:
        return 0.0, True
    variances = [abs(orig_map[d] - new_map[d]) for d in shared]
    mean_var = round(sum(variances) / len(variances), 2)
    reliable = all(v <= 1 for v in variances)
    return mean_var, reliable


def build_report(
    rubric: RubricOutput,
    critic_output: CriticOutput,
    gap_output: GapOutput,
    next_step: NextStep,
    verdict: str,
    consistency_score: float | None = None,
    consistency_reliable: bool | None = None,
) -> EvaluationReport:
    weighted_final = round(
        0.7 * critic_output.overall_score + 0.3 * gap_output.completeness_score, 2
    )
    return EvaluationReport(
        artifact_type=rubric.artifact_type_detected,
        rubric_dimensions=rubric.dimensions,
        dimension_scores=critic_output.scores,
        gaps=gap_output.gaps,
        overall_score=critic_output.overall_score,
        completeness_score=gap_output.completeness_score,
        weighted_final_score=weighted_final,
        summary=critic_output.summary,
        next_best_step=next_step,
        proof_of_work_verdict=verdict,
        consistency_score=consistency_score,
        consistency_reliable=consistency_reliable,
    )