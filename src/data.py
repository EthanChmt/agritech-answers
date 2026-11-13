"""Chargement, nettoyage et découpage des données."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BOOL_COLS, CLEAN_CSV, FEATURES, RAW_CSV, SEED, TARGET, TEST_SIZE


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons et les rendements négatifs (impossibles physiquement)."""
    return df.drop_duplicates().loc[lambda d: d[TARGET] >= 0].reset_index(drop=True)


def load_clean(path=CLEAN_CSV, raw_path=RAW_CSV, save: bool = True) -> pd.DataFrame:
    """Lit le CSV nettoyé ; s'il n'existe pas, le reconstruit depuis le CSV brut."""
    if path.exists():
        return pd.read_csv(path)
    df = clean(load_raw(raw_path))
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare variables explicatives et cible ; les booléens deviennent 0/1."""
    X = df[FEATURES].copy()
    X[BOOL_COLS] = X[BOOL_COLS].astype(int)
    return X, df[TARGET]


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Découpage train / test reproductible (même jeu de test que dans les notebooks)."""
    X, y = make_xy(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed)