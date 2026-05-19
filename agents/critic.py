import json
import asyncio
import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import CriticOutput, DimensionScore, RubricDimension
from utils.llm_client import get_critic_llm
from config import get_settings

CRITIC_SYSTEM_PROMPT = """You are a strict but fair learning evaluator -- a senior expert in your field.

You will receive a learner artifact and a rubric of evaluation dimensions.
Score each dimension on a scale of 1-5 and provide concise, evidence-backed rationale.

Scoring guide:
  5 = Exceptional. Exceeds expectations, demonstrates mastery.
  4 = Strong. Meets expectations with minor gaps.
  3 = Adequate. Core elements present but notable weaknesses.
  2 = Developing. Significant gaps or errors, but effort is visible.
  1 = Insufficient. Does not meet the standard for this dimension.

Rules:
- Be honest and calibrated. Do not inflate scores to be encouraging.
- For each dimension, quote or reference a specific part of the artifact as evidence.
- The "evidence" field must be a direct excerpt or paraphrase from the artifact, not a generic statement.
- Compute overall_score as the weighted average: sum(score * weight) / sum(weight).
- Write a 2-3 sentence "summary" that synthesises performance across all dimensions.

Respond ONLY with valid JSON. No markdown, no explanation.

JSON schema:
{
  "scores": [
    {
      "dimension": "string",
      "score": integer (1-5),
      "rationale": "string (why this score)",
      "evidence": "string (specific excerpt or paraphrase from artifact)"
    }
  ],
  "overall_score": float,
  "summary": "string"
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
        raise ValueError(f"Critic returned non-JSON: {raw[:200]}")


async def run_critic(
    artifact: str,
    dimensions: list[RubricDimension],
    artifact_type: str,
    context: str | None,
    shuffle_dimensions: bool = False,
    **_kwargs,
) -> CriticOutput:
    """
    Score artifact on each rubric dimension.
    Uses full critic LLM (gemini-2.0-flash by default).
    shuffle_dimensions=True is used by the self-eval consistency check.
    """
    settings = get_settings()
    llm = get_critic_llm()

    dims = list(dimensions)
    if shuffle_dimensions:
        import random
        dims = random.sample(dims, len(dims))

    dims_text = "\n".join(
        f"- {d.name} (weight: {d.weight}): {d.description}" for d in dims
    )

    user_content = f"Artifact type: {artifact_type}\n"
    if context:
        user_content += f"Learning objective: {context}\n"
    user_content += f"\nRubric dimensions:\n{dims_text}\n\nArtifact:\n{artifact[:4000]}"

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=settings.agent_timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Critic agent timed out.")

    raw = response.content.strip()
    data = _parse_json(raw)

    scores = [DimensionScore(**s) for s in data.get("scores", [])]

    # Recalculate weighted average ourselves for correctness
    dim_map = {d.name: d.weight for d in dimensions}
    weighted_sum = sum(s.score * dim_map.get(s.dimension, 1.0) for s in scores)
    weight_total = sum(dim_map.get(s.dimension, 1.0) for s in scores)
    overall = round(weighted_sum / weight_total, 2) if weight_total else 0.0

    return CriticOutput(
        scores=scores,
        overall_score=overall,
        summary=data.get("summary", ""),
    )