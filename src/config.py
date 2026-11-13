"""Constantes partagées par les notebooks, le script d'entraînement et (si besoin) l'API."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Données
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Agriculture_Crop_Yield" / "crop_yield.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "crop_yield_clean.csv"

TARGET = "Yield_tons_per_hectare"
NUM_COLS = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
BOOL_COLS = ["Fertilizer_Used", "Irrigation_Used"]
CAT_COLS = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
FEATURES = NUM_COLS + BOOL_COLS + CAT_COLS
CROPS = ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]

# --- Reproductibilité
SEED = 42
TEST_SIZE = 0.2

# --- MLflow
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "crop-yield-optimisation"
REGISTRY_NAME = "crop-yield-regressor"
CHAMPION_ALIAS = "champion"

# --- Protocole d'optimisation (identique au notebook 03)
MODEL_NAMES = ["ridge", "hist-gradient-boosting"]
SIMPLE_MODEL = "ridge"
TOLERANCE_SELECTION = 0.001  # t/ha : on préfère le modèle simple s'il est à moins de 0,001 du meilleur RMSE
N_TUNE_SAMPLE = 100_000
CV_FOLDS = 5
N_TRIALS = 40