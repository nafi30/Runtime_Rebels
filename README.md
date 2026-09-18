# GridWise Energy Optimizer

**Team:** Runtime_Rebels | **Event:** BUP CSE Fest 2026 Hackathon

GridWise is an API-based energy scheduling system built to minimize daily electricity costs. It uses linear programming for optimal battery dispatch and parses natural language constraints using LLMs.

## Architecture Highlights
* **Mathematical Optimization:** Uses `scipy.optimize.linprog` (HiGHS solver) for deterministic, lowest-cost 24-hour schedules.
* **NLP Extraction:** Uses Gemini 1.5 Flash (via OpenAI SDK compatibility) to translate operator notes into strict mathematical constraints.
* **Speed & Validation:** Built on FastAPI and Pydantic for low latency and schema compliance.
* **Fallback Resiliency:** Includes an offline heuristic NLP fallback to prevent API failures during LLM provider downtime.

## Tech Stack
* Python 3.11+
* FastAPI & Uvicorn
* SciPy & NumPy
* Pydantic v2
* Google Gemini API

## Quickstart (Docker)
To run the containerized application locally:
```bash
docker build -t gridwise-api .
docker run -p 8000:8000 -e OPENAI_API_KEY="your_gemini_key" -e OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" -e LLM_MODEL="gemini-1.5-flash" gridwise-api
```

## API Endpoints
### GET /health
Returns system status.
```json
{ "status": "ok" }
```

### POST /optimize-energy
Accepts a 24-hour scenario with demand, solar forecasts, grid tariffs, battery specs, and operator notes. Returns the cost-minimized hourly dispatch plan and parsed directives.

## Running Local Tests
The test suite validates the official BUP sample cases.
```bash
pip install -r requirements.txt
python test_samples.py
```