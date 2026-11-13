import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from api.main import ModelArtifacts, app, get_artifacts, load_artifacts
from api.schemas import CROPS

NUM_COLS = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
BOOL_COLS = ["Fertilizer_Used", "Irrigation_Used"]
CAT_COLS = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
TEST_RMSE = 0.50

VALID_CONTEXT = {
    "rainfall_mm": 600,
    "temperature_celsius": 25,
    "fertilizer_used": True,
    "irrigation_used": True,
}


def make_training_frame(n=6000, seed=42):
    """Jeu synthétique reproduisant la structure du jeu réel : la culture n'a aucun effet."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "Rainfall_mm": rng.uniform(100, 1000, n),
        "Temperature_Celsius": rng.uniform(15, 40, n),
        "Days_to_Harvest": rng.integers(60, 150, n),
        "Fertilizer_Used": rng.integers(0, 2, n),
        "Irrigation_Used": rng.integers(0, 2, n),
        "Region": rng.choice(["East", "North", "South", "West"], n),
        "Soil_Type": rng.choice(["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"], n),
        "Crop": rng.choice(CROPS, n),
        "Weather_Condition": rng.choice(["Cloudy", "Rainy", "Sunny"], n),
    })
    target = (
        0.005 * frame["Rainfall_mm"] + 0.02 * frame["Temperature_Celsius"]
        + 1.5 * frame["Fertilizer_Used"] + 1.2 * frame["Irrigation_Used"]
        + rng.normal(0, 0.5, n)
    )
    return frame, target


def make_pipeline():
    X, y = make_training_frame()
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("bool", "passthrough", BOOL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=0.3))]).fit(X, y)


class CropDependentStub:
    """Faux pipeline dont la prédiction dépend fortement de la culture (écart de 5 t/ha)."""

    def predict(self, frame):
        return frame["Crop"].map({crop: float(i) for i, crop in enumerate(CROPS)}).to_numpy()


class FailingStub:
    def predict(self, frame):
        raise ValueError("colonnes manquantes")


@pytest.fixture(scope="session")
def pipeline():
    return make_pipeline()


@pytest.fixture
def client(pipeline):
    """Client dont le modèle est le pipeline synthétique."""
    app.dependency_overrides[get_artifacts] = lambda: ModelArtifacts(pipeline=pipeline, rmse=TEST_RMSE)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def stub_client():
    """Client dont la prédiction dépend fortement de la culture."""
    app.dependency_overrides[get_artifacts] = lambda: ModelArtifacts(pipeline=CropDependentStub(), rmse=TEST_RMSE)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def failing_client():
    app.dependency_overrides[get_artifacts] = lambda: ModelArtifacts(pipeline=FailingStub(), rmse=TEST_RMSE)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def saved_model(tmp_path, pipeline, monkeypatch):
    """Pipeline et métadonnées écrits sur disque, comme l'export du notebook 03."""
    model_path = tmp_path / "final_model.joblib"
    metadata_path = tmp_path / "final_model_metadata.json"
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps({"test_rmse": 0.4993}), encoding="utf-8")
    monkeypatch.setattr("api.main.MODEL_PATH", model_path)
    monkeypatch.setattr("api.main.METADATA_PATH", metadata_path)
    load_artifacts.cache_clear()
    yield model_path
    load_artifacts.cache_clear()