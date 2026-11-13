import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from api.schemas import (
    CROPS,
    ContextRequest,
    CropPrediction,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)

logger = logging.getLogger("crop-yield-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/final_model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", MODEL_PATH.with_name("final_model_metadata.json")))
DEFAULT_RMSE = 0.50

app = FastAPI(
    title="Crop Yield Prediction & Recommendation API",
    version="1.0.0",
)


@dataclass
class ModelArtifacts:
    pipeline: object
    rmse: float


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts | None:
    """Charge le pipeline sauvegardé et le RMSE de test associé."""
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Chargement du modèle impossible : %s", MODEL_PATH)
        return None
    rmse = DEFAULT_RMSE
    if METADATA_PATH.exists():
        rmse = float(json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("test_rmse", DEFAULT_RMSE))
    return ModelArtifacts(pipeline=pipeline, rmse=rmse)


def get_artifacts() -> ModelArtifacts:
    artifacts = load_artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur.")
    return artifacts


def build_frame(context: ContextRequest, crops: list[str]) -> pd.DataFrame:
    """Une ligne par culture, avec les colonnes attendues par le pipeline d'entraînement."""
    return pd.DataFrame({
        "Rainfall_mm": context.rainfall_mm,
        "Temperature_Celsius": context.temperature_celsius,
        "Days_to_Harvest": context.days_to_harvest,
        "Fertilizer_Used": int(context.fertilizer_used),
        "Irrigation_Used": int(context.irrigation_used),
        "Region": context.region,
        "Soil_Type": context.soil_type,
        "Crop": crops,
        "Weather_Condition": context.weather_condition,
    })


def predict(artifacts: ModelArtifacts, frame: pd.DataFrame):
    try:
        return artifacts.pipeline.predict(frame)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {exc}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": load_artifacts() is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_yield(data: PredictionRequest, artifacts: ModelArtifacts = Depends(get_artifacts)):
    value = float(predict(artifacts, build_frame(data, [data.crop]))[0])
    return PredictionResponse(
        crop=data.crop,
        predicted_yield=round(value, 3),
        lower_bound=round(max(value - artifacts.rmse, 0.0), 3),
        upper_bound=round(value + artifacts.rmse, 3),
        rmse=artifacts.rmse,
        message=(
            f"Rendement estimé à {value:.2f} t/ha, avec une erreur typique de "
            f"{artifacts.rmse:.2f} t/ha."
        ),
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_crop(data: ContextRequest, artifacts: ModelArtifacts = Depends(get_artifacts)):
    predictions = predict(artifacts, build_frame(data, CROPS))
    ranked = sorted(zip(CROPS, predictions), key=lambda item: item[1], reverse=True)

    ranking = [
        CropPrediction(
            rank=position,
            crop=crop,
            predicted_yield=round(float(value), 3),
            lower_bound=round(max(float(value) - artifacts.rmse, 0.0), 3),
            upper_bound=round(float(value) + artifacts.rmse, 3),
        )
        for position, (crop, value) in enumerate(ranked, start=1)
    ]
    spread = float(ranked[0][1] - ranked[-1][1])
    warning = spread < artifacts.rmse

    if warning:
        message = (
            f"L'écart entre la meilleure et la moins bonne culture ({spread:.3f} t/ha) est inférieur "
            f"à l'incertitude du modèle (±{artifacts.rmse:.2f} t/ha) : ce classement ne permet pas de "
            "désigner une culture optimale. Le rendement dépend surtout de la pluie, de l'engrais et de l'irrigation."
        )
    else:
        message = "Les écarts entre cultures dépassent l'incertitude du modèle."

    return RecommendationResponse(
        ranking=ranking,
        rmse=artifacts.rmse,
        spread=round(spread, 4),
        uncertainty_warning=warning,
        message=message,
    )