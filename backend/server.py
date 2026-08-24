import logging
from contextlib import asynccontextmanager
from typing import List
 
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")
 
MODEL_PATH = "model.pkl"          # your trained model artifact
SCALER_PATH = "scaler.pkl"        # optional: fitted preprocessing objects
FEATURE_ORDER = [                 # must match training-time column order
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
]
 
ml_artifacts = {}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    ml_artifacts["model"] = joblib.load(MODEL_PATH)
    try:
        ml_artifacts["scaler"] = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        ml_artifacts["scaler"] = None
        logger.warning("No scaler found, skipping scaling step.")
    logger.info("Model artifacts loaded.")
    yield
    ml_artifacts.clear()
 
 
app = FastAPI(title="Model Inference API", version="1.0.0", lifespan=lifespan)
 
 

class PredictionRequest(BaseModel):
    feature_1: float = Field(..., description="Description of feature_1")
    feature_2: float = Field(..., description="Description of feature_2")
    feature_3: float = Field(..., description="Description of feature_3")
    feature_4: str = Field(..., description="Categorical feature, e.g. 'A'/'B'/'C'")
 
 
class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]
 
 
class PredictionResponse(BaseModel):
    prediction: float
    probability: float | None = None
 
 
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
 

def preprocess(records: List[PredictionRequest]) -> np.ndarray:
    df = pd.DataFrame([r.model_dump() for r in records])
 
    # Example: one-hot encode a categorical column the same way training did
    df = pd.get_dummies(df, columns=["feature_4"])
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER]
 
    if ml_artifacts["scaler"] is not None:
        return ml_artifacts["scaler"].transform(df)
    return df.to_numpy()
 
 
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_artifacts.get("model") is not None}
 
 
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        X = preprocess([request])
        model = ml_artifacts["model"]
        pred = model.predict(X)[0]
 
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0].max())
 
        return PredictionResponse(prediction=float(pred), probability=prob)
 
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
 
 
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        X = preprocess(request.records)
        model = ml_artifacts["model"]
        preds = model.predict(X)
 
        probs = [None] * len(preds)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X).max(axis=1).tolist()
 
        results = [
            PredictionResponse(prediction=float(p), probability=probs[i])
            for i, p in enumerate(preds)
        ]
        return BatchPredictionResponse(predictions=results)
 
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
 
