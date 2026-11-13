"""Prétraitement et construction des modèles. Un seul endroit, utilisé partout."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BOOL_COLS, CAT_COLS, NUM_COLS, SEED

MODEL_FACTORIES = {
    "dummy-mean": lambda **p: DummyRegressor(**p),
    "linear": lambda **p: LinearRegression(**p),
    "ridge": lambda **p: Ridge(random_state=SEED, **p),
    "random-forest": lambda **p: RandomForestRegressor(random_state=SEED, n_jobs=-1, **p),
    "hist-gradient-boosting": lambda **p: HistGradientBoostingRegressor(random_state=SEED, **p),
}


def make_preprocessor() -> ColumnTransformer:
    """Numériques standardisées, booléens en 0/1, catégorielles en one-hot (catégories inconnues ignorées)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("bool", "passthrough", BOOL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(name: str, params: dict | None = None) -> Pipeline:
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Modèle inconnu : {name!r}. Choix : {sorted(MODEL_FACTORIES)}")
    return Pipeline([("preprocess", make_preprocessor()), ("model", MODEL_FACTORIES[name](**(params or {})))])