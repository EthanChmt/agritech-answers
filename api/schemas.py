from typing import Literal

from pydantic import BaseModel, Field

Crop = Literal["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]
Region = Literal["East", "North", "South", "West"]
SoilType = Literal["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"]
Weather = Literal["Cloudy", "Rainy", "Sunny"]

CROPS: list[str] = ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]


class ContextRequest(BaseModel):
    """Conditions de la parcelle. Les bornes sont celles du jeu d'entraînement."""

    rainfall_mm: float = Field(..., ge=100, le=1000, description="Pluviométrie en mm")
    temperature_celsius: float = Field(..., ge=15, le=40, description="Température moyenne en °C")
    fertilizer_used: bool = Field(..., description="Engrais utilisé")
    irrigation_used: bool = Field(..., description="Irrigation utilisée")
    region: Region = Field("North", description="Région (sans effet mesurable sur le rendement)")
    soil_type: SoilType = Field("Loam", description="Type de sol (sans effet mesurable sur le rendement)")
    weather_condition: Weather = Field("Sunny", description="Météo (sans effet mesurable sur le rendement)")
    days_to_harvest: int = Field(105, ge=60, le=150, description="Jours avant récolte (sans effet mesurable sur le rendement)")


class PredictionRequest(ContextRequest):
    crop: Crop = Field(..., description="Culture")


class PredictionResponse(BaseModel):
    crop: str
    predicted_yield: float = Field(..., description="Rendement prédit en t/ha")
    lower_bound: float = Field(..., description="Prédiction moins le RMSE du modèle (t/ha, minimum 0)")
    upper_bound: float = Field(..., description="Prédiction plus le RMSE du modèle (t/ha)")
    rmse: float = Field(..., description="Erreur typique du modèle sur le jeu de test (t/ha)")
    message: str


class CropPrediction(BaseModel):
    rank: int
    crop: str
    predicted_yield: float
    lower_bound: float
    upper_bound: float


class RecommendationResponse(BaseModel):
    ranking: list[CropPrediction] = Field(..., description="Cultures triées par rendement prédit décroissant")
    rmse: float = Field(..., description="Erreur typique du modèle sur le jeu de test (t/ha)")
    spread: float = Field(..., description="Écart entre la meilleure et la moins bonne culture (t/ha)")
    uncertainty_warning: bool = Field(..., description="Vrai si l'écart entre cultures est inférieur au RMSE")
    message: str