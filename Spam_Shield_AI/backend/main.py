import io
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import load_or_train_model, predict_messages

app = FastAPI(title="Spam Message Classifier API", version="1.0.0",
              description="Machine-learning API for spam message classification.")

# For production, set FRONTEND_ORIGIN to the deployed Netlify site URL.
import os
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
origins = ["*"] if frontend_origin == "*" else [x.strip() for x in frontend_origin.split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

_model = None
_metrics = None
_dataset_source = None

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)

@app.on_event("startup")
def startup():
    global _model, _metrics, _dataset_source
    _model, _metrics, _dataset_source = load_or_train_model()

@app.get("/")
def root():
    return {"service": "Spam Message Classifier API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/metrics")
def metrics():
    if _metrics is None:
        raise HTTPException(503, "Model is not ready.")
    return _metrics

@app.post("/predict")
def predict(request: MessageRequest):
    if _model is None:
        raise HTTPException(503, "Model is not ready.")
    result = predict_messages(_model, [request.message])[0]
    return result

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model is not ready.")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    raw = await file.read()
    if len(raw) > 5_000_000:
        raise HTTPException(413, "CSV is too large. Maximum file size is 5 MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, "Could not read CSV. Please check its format.") from exc
    column = "message" if "message" in df.columns else ("text" if "text" in df.columns else None)
    if column is None:
        raise HTTPException(400, "CSV must contain a 'message' or 'text' column.")
    if len(df) == 0:
        raise HTTPException(400, "CSV has no rows.")
    if len(df) > 10000:
        raise HTTPException(413, "CSV can contain at most 10,000 messages.")
    messages = df[column].fillna("").astype(str).tolist()
    predictions = predict_messages(_model, messages)
    results = [{"message": msg, **result} for msg, result in zip(messages, predictions)]
    return {"total": len(results), "spam_count": sum(r["label"] == "Spam" for r in results),
            "not_spam_count": sum(r["label"] == "Not Spam" for r in results), "results": results}
