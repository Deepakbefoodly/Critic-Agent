import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import get_settings
from schemas import (
    EvaluateRequest,
    EvaluationReport,
    ErrorResponse,
    HealthResponse,
)
from utils.pipeline import run_evaluation_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set. Requests will fail.")
    logger.info(
        "Starting flo101 Critic Agent | critic=%s fast=%s",
        settings.critic_model,
        settings.fast_model,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="flo101 Critic Agent -- Proof-of-Work Evaluator",
    description=(
        "Multi-agent evaluation pipeline powered by LangChain + Gemini. "
        "Rubric builder -> critic + gap detector (parallel) -> synthesis."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation failed",
            code="VALIDATION_ERROR",
            detail=str(exc),
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(status="ok", version="2.0.0")


@app.post(
    "/evaluate",
    response_model=EvaluationReport,
    tags=["Evaluation"],
    summary="Evaluate a learner artifact",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or insufficient artifact"},
        408: {"model": ErrorResponse, "description": "Agent timeout"},
        429: {"model": ErrorResponse, "description": "LLM rate limit"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
)
async def evaluate(request: EvaluateRequest):
    """
    Run the full Critic Agent & Proof-of-Work evaluation pipeline.

    Powered by LangChain with Google Gemini as the LLM provider.
    - Fast agents (rubric builder, gap detector): gemini-2.0-flash-lite
    - Quality agents (critic, synthesis): gemini-2.0-flash
    """
    try:
        return await run_evaluation_pipeline(request)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))

    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "rate" in err or "429" in err:
            raise HTTPException(status_code=429, detail="Gemini rate limit reached. Retry in a moment.")
        if "api_key" in err or "credential" in err or "401" in err:
            raise HTTPException(status_code=401, detail="Invalid GOOGLE_API_KEY.")
        logger.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline error: {e}")