import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic.main import BaseModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

model = None


class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionOutput(BaseModel):
    prediction: int
    class_name: str
    confidence: float


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully")


def clean_model():
    global model
    model = None
    print("Model cleaned successfully")


@asynccontextmanager
async def life_span(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Iris classification API", version="1.0.0", lifespan=life_span)


@app.get("/")
async def root():
    return {
        "message": "Iris classification API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict"},
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = np.array(
        [[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]]
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    class_names = ["setosa", "versicolor", "virginica"]

    return PredictionOutput(
        prediction=prediction, class_name=class_names[prediction], confidence=confidence
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
