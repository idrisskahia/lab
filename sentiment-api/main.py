from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline
import os

app = FastAPI(title="Sentiment API", version="0.1.0")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")

class PredictResponse(BaseModel):
    label: str
    score: float
    model_name: str

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
HF_MODEL = os.getenv("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

nlp: Pipeline | None = None
model_name_loaded: str = ""

@app.on_event("startup")
def load_model():
    """
    Load from local ./model if present; otherwise fall back to Hugging Face Hub.
    This keeps local/dev fast and makes Docker builds deterministic later.
    """
    global nlp, model_name_loaded
    try:
        if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
            nlp = pipeline("sentiment-analysis", model=MODEL_DIR)
            model_name_loaded = f"local:{MODEL_DIR}"
        else:
            nlp = pipeline("sentiment-analysis", model=HF_MODEL)
            model_name_loaded = HF_MODEL
    except Exception as e:
        # Fail loudly at startup so we don't accept traffic without a model
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name_loaded}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not nlp:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        out = nlp(req.text)[0]  # {'label': 'POSITIVE', 'score': 0.999...}
        return PredictResponse(label=out["label"], score=float(out["score"]), model_name=model_name_loaded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Welcome to Sentiment API. POST /predict with {'text': '...'}",
        "docs": "/docs",
        "health": "/health",
        "model": model_name_loaded,
    }

