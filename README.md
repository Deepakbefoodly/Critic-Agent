# flo101 — Critic Agent + Proof-of-Work Evaluator

A multi-agent FastAPI service that evaluates learner artifacts against an inferred rubric, identifies gaps, and produces a proof-of-work verdict.

**Powered by:** LangChain + Google Gemini

---

## What it does

Submit any learner artifact — code, a brief, a draft, an essay — and receive:

- **Rubric-based scores** across 3–5 dimensions inferred for that artifact type
- **Dimension-level rationale** with evidence from the artifact
- **Gap analysis** — what's *missing*, not just what's weak
- **Weighted final score** (70% quality + 30% completeness)
- **Proof-of-work verdict**: `demonstrated` / `partial` / `not demonstrated`
- **Next best step**: the single highest-leverage improvement the learner should make
- **Consistency score**: self-evaluation that re-runs scoring with shuffled criteria to flag unreliable AI output

---

## Architecture

```
POST /evaluate
       │
       ▼
┌─────────────────────┐
│   Rubric Builder    │  gemini-2.0-flash-lite — infers 3–5 dimensions
└──────────┬──────────┘
           │
   ┌───────┴──────────────────────────────────┐
   │                                          │
   ▼                                          ▼
┌──────────────────┐              ┌───────────────────────┐
│  Critic Agent    │ ←PARALLEL→  │   Gap Detector        │
│  gemini-2.0-flash│             │   gemini-2.0-flash-lite│
│  Score each dim  │             │   Find what's missing  │
└────────┬─────────┘             └───────────┬────────────┘
         └─────────────┬─────────────────────┘
                       │
          ┌────────────┴──────────────────────┐
          │                                   │
          ▼                                   ▼
┌──────────────────────┐       ┌──────────────────────────┐
│  Synthesis Agent     │←PAR→ │  Self-eval (critic run   │
│  gemini-2.0-flash    │       │  with shuffled criteria) │
│  Verdict + next step │       └──────────────────────────┘
└──────────┬───────────┘
           ▼
    EvaluationReport (JSON)
```

### Agent model assignment

| Agent | Model | Reason |
|---|---|---|
| Rubric builder | `gemini-2.0-flash-lite` | Structural inference, not evaluative |
| Gap detector | `gemini-2.0-flash-lite` | Pattern matching; cheap + fast |
| Critic | `gemini-2.0-flash` | Core quality signal — use best available |
| Synthesiser | `gemini-2.0-flash` | Verdict accuracy matters |
| Self-eval | `gemini-2.0-flash` | Must match critic quality for valid comparison |

Override any model via env vars (see Configuration below).

---

## Quick start

### Prerequisites
- Python 3.11+
- A Google Gemini API key

### Setup

```bash
git clone <repo>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
uvicorn main:app --reload
```

Server: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

---

## API reference

### `POST /evaluate`

**Request**

```json
{
  "artifact": "string (required, min 30 chars)",
  "artifact_type": "auto | code | brief | draft | essay  (default: auto)",
  "context": "string (optional — learning objective or task description)"
}
```

**Response: `EvaluationReport`**

```json
{
  "artifact_type": "code",
  "rubric_dimensions": [
    { "name": "Correctness", "description": "...", "weight": 2.0 }
  ],
  "dimension_scores": [
    {
      "dimension": "Correctness",
      "score": 4,
      "rationale": "The algorithm correctly handles the base case...",
      "evidence": "while left <= right: ..."
    }
  ],
  "gaps": [
    {
      "area": "No unit tests",
      "description": "There are no test cases to verify correctness.",
      "severity": "moderate"
    }
  ],
  "overall_score": 3.8,
  "completeness_score": 3,
  "weighted_final_score": 3.56,
  "summary": "Solid implementation with clean logic...",
  "next_best_step": {
    "action": "Add 3 unit tests covering empty array, single element, and not-found cases.",
    "rationale": "Tests are the most critical missing element for a production-ready submission.",
    "expected_impact": "Completeness score 3 → 5, weighted_final 3.56 → 4.1"
  },
  "proof_of_work_verdict": "partial",
  "consistency_score": 0.33,
  "consistency_reliable": true
}
```

**Proof-of-work verdict logic**

| Verdict | Condition |
|---|---|
| `demonstrated` | overall ≥ 3.5 AND completeness ≥ 3 AND no critical gaps |
| `partial` | overall 2.5–3.4 OR 1 critical gap with real effort |
| `not demonstrated` | overall < 2.5 OR 2+ critical gaps |

**Consistency score**
Re-runs the critic with shuffled rubric order. `consistency_reliable: false` means a dimension varied by >1 point — treat the result with caution.

### `GET /health`

```json
{ "status": "ok", "version": "2.0.0" }
```

---

## Running tests

```bash
pytest tests/ -v
```

---