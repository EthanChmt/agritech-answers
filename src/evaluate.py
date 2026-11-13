"""Métriques et diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CROPS, SEED


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(pipe, X, y) -> dict[str, float]:
    pred = pipe.predict(X)
    return {"rmse": rmse(y, pred), "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def crop_effect(pipe, X: pd.DataFrame, n: int = 2000, seed: int = SEED, crops=CROPS) -> pd.DataFrame:
    """Pour n contextes, prédit le rendement de chaque culture (contexte identique).

    L'écart max - min entre cultures, comparé au RMSE, dit si la recommandation a un sens.
    """
    ctx = X.sample(min(n, len(X)), random_state=seed).reset_index(drop=True)
    return pd.DataFrame({crop: pipe.predict(ctx.assign(Crop=crop)) for crop in crops})