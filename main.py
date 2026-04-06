"""
main.py — FastAPI Backend for Sentiment Analysis UI
Serves the web UI and API endpoints. All ML logic lives in model.py.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from model import SentimentModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Model instance ────────────────────────────────────────────────────────────
sentiment_model = SentimentModel()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Analysis UI")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("startup")
async def startup_event():
    """Load or train the model on server startup."""
    sentiment_model.load_or_train()


# ── Request / Response models ─────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    positive_prob: float
    negative_prob: float
    processed_text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: ReviewRequest):
    """Predict sentiment for a restaurant review."""
    try:
        result = sentiment_model.predict(req.text)
        return PredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    """Return current model performance metrics."""
    if not sentiment_model.trained:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    return JSONResponse(content=sentiment_model.metrics)


@app.get("/api/samples")
async def get_samples():
    """Return sample reviews from the dataset."""
    return JSONResponse(content=sentiment_model.sample_reviews)


@app.post("/api/retrain")
async def retrain():
    """Retrain the model from scratch."""
    try:
        metrics = sentiment_model.retrain()
        return JSONResponse(content={"status": "success", "metrics": metrics})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
