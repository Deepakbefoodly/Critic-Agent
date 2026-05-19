import json
import asyncio
import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import GapOutput, Gap
from utils.llm_client import get_fast_llm
from config import get_settings

GAP_SYSTEM_PROMPT = """You are a learning quality assurance expert. Your role is NOT to score what's present --
that's handled by another agent. Your role is to identify what is MISSING, OVERLOOKED, or NOT ADDRESSED.

Think adversarially: if a senior reviewer read this, what would immediately concern them?
What assumptions are made but not stated? What edge cases are ignored? What critical requirements are absent?

Severity guide:
  critical = A professional would reject this work for this reason alone.
  moderate = Noticeable gap that reduces credibility or usefulness.
  minor = Nice-to-have; absence doesn't break the work.

Rules:
- Only flag genuine absences -- not things that are weak (the critic scores weakness).
- Be specific. "Missing error handling" is better than "incomplete".
- 2-5 gaps is the right range. 0 means fully complete; 6+ is nitpicking.
- completeness_score: 1 = severely incomplete, 5 = very complete.

Respond ONLY with valid JSON. No markdown, no explanation.

JSON schema:
{
  "gaps": [
    {
      "area": "string (short label)",
      "description": "string (what is missing and why it matters)",
      "severity": "critical|moderate|minor"
    }
  ],
  "completeness_score": integer (1-5)
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
        raise ValueError(f"Gap detector returned non-JSON: {raw[:200]}")


async def detect_gaps(
    artifact: str,
    artifact_type: str,
    context: str | None,
) -> GapOutput:
    """
    Identify what's missing. Runs in parallel with the critic.
    """
    settings = get_settings()
    llm = get_fast_llm()

    user_content = f"Artifact type: {artifact_type}\n"
    if context:
        user_content += f"Learning objective / task context: {context}\n"
    user_content += f"\nArtifact:\n{artifact[:3000]}"

    messages = [
        SystemMessage(content=GAP_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=settings.agent_timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Gap detector agent timed out.")

    raw = response.content.strip()
    data = _parse_json(raw)

    return GapOutput(
        gaps=[Gap(**g) for g in data.get("gaps", [])],
        completeness_score=data.get("completeness_score", 3),
    )


def get_fallback_gap_output() -> GapOutput:
    return GapOutput(
        gaps=[
            Gap(
                area="Gap analysis unavailable",
                description="The gap detection agent encountered an error. Manual review recommended.",
                severity="minor",
            )
        ],
        completeness_score=3,
    )