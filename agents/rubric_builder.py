import json
import asyncio
import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import RubricOutput, RubricDimension, ArtifactType
from utils.llm_client import get_fast_llm
from config import get_settings

RUBRIC_SYSTEM_PROMPT = """You are an expert learning evaluator. Your job is to build a rubric for evaluating a learner's artifact.

Given an artifact and its type, output a JSON object with the evaluation dimensions most relevant to that artifact type.

Rules:
- Produce exactly 3-5 dimensions. Fewer than 3 means the rubric is too thin; more than 5 is too granular.
- Each dimension must be distinct -- no overlap in what it measures.
- Weights: 1.0 = normal, 2.0 = critical for this artifact type, 0.5 = nice-to-have.
- If artifact_type is "auto", infer the type from content.

Respond ONLY with valid JSON. No markdown, no explanation, no code fences.

JSON schema:
{
  "artifact_type_detected": "string (code|brief|draft|essay|other)",
  "dimensions": [
    {
      "name": "string",
      "description": "string (what exactly is being measured)",
      "weight": number
    }
  ]
}

Examples by type:
- code: correctness, code quality, edge case handling, efficiency, documentation
- brief: clarity of argument, supporting evidence, structure, actionability, conciseness
- essay: thesis strength, argument coherence, evidence use, critical thinking, writing quality
- draft: core idea clarity, completeness, logical flow, language precision, audience fit
"""


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from response: {raw[:200]}")


async def build_rubric(
    artifact: str,
    artifact_type: ArtifactType,
    context: str | None,
    **_kwargs,
) -> RubricOutput:
    settings = get_settings()
    llm = get_fast_llm()

    user_content = f"Artifact type hint: {artifact_type.value}\n"
    if context:
        user_content += f"Learning objective: {context}\n"
    user_content += f"\nArtifact:\n{artifact[:3000]}"

    messages = [
        SystemMessage(content=RUBRIC_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=settings.agent_timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Rubric builder agent timed out. Using fallback rubric.")

    raw = response.content.strip()
    data = _parse_json(raw)

    rubric = RubricOutput(
        artifact_type_detected=data.get("artifact_type_detected", artifact_type.value),
        dimensions=[RubricDimension(**d) for d in data.get("dimensions", [])],
    )

    if len(rubric.dimensions) < 2:
        raise ValueError(
            "The rubric builder generated fewer than 2 dimensions. "
            "The artifact may be too short or too vague to evaluate."
        )

    return rubric


def get_fallback_rubric(artifact_type: ArtifactType) -> RubricOutput:
    return RubricOutput(
        artifact_type_detected=artifact_type.value,
        dimensions=[
            RubricDimension(
                name="Clarity",
                description="How clearly the main idea or intent is communicated",
                weight=1.5,
            ),
            RubricDimension(
                name="Completeness",
                description="Whether all expected components are present",
                weight=1.5,
            ),
            RubricDimension(
                name="Quality of reasoning",
                description="Soundness of logic, structure, or implementation",
                weight=2.0,
            ),
        ],
    )