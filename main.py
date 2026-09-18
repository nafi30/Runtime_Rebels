import logging
from fastapi import FastAPI, HTTPException

from schemas import OptimizeRequest, OptimizeResponse
from guardrails import validate_and_sanitize_directives
from optimizer import solve_24h_schedule
from interpreter import extract_directives_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gridwise.main")

app = FastAPI(title="GridWise Optimizer")


@app.get("/")
def root():
    """Root landing endpoint with system status and links."""
    return {
        "service": "GridWise Optimizer API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "optimize": "POST /optimize-energy",
    }


@app.get("/health")
def health_check():
    """Health check probe returning HTTP 200 within 60s of startup."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(request: OptimizeRequest):
    """
    Receives 24-hour campus energy forecast and natural-language operator notes,
    extracts structured directives via LLM, validates them deterministically via guardrails,
    and computes the cost-optimal 24-hour schedule via SciPy HiGHS LP solver.
    """
    try:
        # Step 1: LLM Extraction
        raw_directives = extract_directives_llm(
            operator_notes=request.operator_notes,
            battery_capacity=request.battery.capacity_kwh,
        )

        # Step 2: Deterministic Guardrails & Sanitization
        safe_directives = validate_and_sanitize_directives(
            raw_directives=raw_directives,
            operator_notes=request.operator_notes,
            battery_capacity=request.battery.capacity_kwh,
        )

        # Step 3: HiGHS Mathematical LP Scheduler
        final_plan = solve_24h_schedule(
            request=request,
            directives=safe_directives,
        )

        # Step 4: Return optimal response
        return final_plan

    except Exception as e:
        logger.error(f"Error executing energy optimization pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
