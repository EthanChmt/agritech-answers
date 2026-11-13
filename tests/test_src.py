"""Tests de src/ : uniquement des données synthétiques (data/raw n'est pas versionné)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean, make_xy, split
from src.evaluate import crop_effect, evaluate, rmse
from src.pipeline import build_pipeline, make_preprocessor
from src.train import select_model


def synthetic_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Region": rng.choice(["East", "North", "South", "West"], n),
        "Soil_Type": rng.choice(["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"], n),
        "Crop": rng.choice(config.CROPS, n),
        "Rainfall_mm": rng.uniform(100, 1000, n),
        "Temperature_Celsius": rng.uniform(15, 40, n),
        "Fertilizer_Used": rng.choice([True, False], n),
        "Irrigation_Used": rng.choice([True, False], n),
        "Weather_Condition": rng.choice(["Cloudy", "Rainy", "Sunny"], n),
        "Days_to_Harvest": rng.integers(60, 150, n),
    })
    df[config.TARGET] = (
        0.005 * df["Rainfall_mm"] + 1.5 * df["Fertilizer_Used"] + 1.2 * df["Irrigation_Used"]
        + rng.normal(0, 0.5, n)
    )
    return df


class TestClean:
    def test_removes_negative_yields_and_duplicates(self):
        df = synthetic_frame(100)
        df.loc[0, config.TARGET] = -1.0
        dirty = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        cleaned = clean(dirty)
        assert (cleaned[config.TARGET] >= 0).all()
        assert len(cleaned) == 99  # 101 lignes - 1 doublon - 1 négatif

    def test_keeps_zero_yield(self):
        df = synthetic_frame(10)
        df.loc[0, config.TARGET] = 0.0
        assert len(clean(df)) == 10


class TestSplit:
    def test_split_is_reproducible(self):
        df = synthetic_frame()
        a, b = split(df), split(df)
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_series_equal(a[3], b[3])

    def test_no_overlap_and_expected_sizes(self):
        X_train, X_test, _, _ = split(synthetic_frame())
        assert len(X_test) == pytest.approx(0.2 * 2000, abs=1)
        assert X_train.index.intersection(X_test.index).empty

    def test_booleans_become_integers(self):
        X, _ = make_xy(synthetic_frame(50))
        assert set(X[config.BOOL_COLS].stack().unique()) <= {0, 1}
        assert list(X.columns) == config.FEATURES


class TestPipeline:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Modèle inconnu"):
            build_pipeline("reseau-de-neurones")

    @pytest.mark.parametrize("name", ["dummy-mean", "linear", "ridge"])
    def test_pipeline_fits_and_predicts(self, name):
        X_train, X_test, y_train, y_test = split(synthetic_frame())
        pipe = build_pipeline(name).fit(X_train, y_train)
        assert pipe.predict(X_test).shape == (len(X_test),)

    def test_ignores_unseen_category(self):
        X_train, X_test, y_train, _ = split(synthetic_frame())
        pipe = build_pipeline("ridge").fit(X_train, y_train)
        assert np.isfinite(pipe.predict(X_test.head(3).assign(Region="Centre"))).all()

    def test_preprocessor_expected_output_width(self):
        X_train, _, _, _ = split(synthetic_frame())
        out = make_preprocessor().fit_transform(X_train)
        # 3 numériques + 2 booléens + 4 régions + 6 sols + 6 cultures + 3 météos
        assert out.shape[1] == 3 + 2 + 4 + 6 + 6 + 3

    def test_params_are_forwarded(self):
        assert build_pipeline("ridge", {"alpha": 7.0}).named_steps["model"].alpha == 7.0


class TestEvaluate:
    def test_rmse_of_perfect_prediction_is_zero(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0

    def test_metrics_on_learnable_data(self):
        X_train, X_test, y_train, y_test = split(synthetic_frame(4000))
        metrics = evaluate(build_pipeline("ridge").fit(X_train, y_train), X_test, y_test)
        assert set(metrics) == {"rmse", "mae", "r2"}
        assert metrics["rmse"] == pytest.approx(0.5, abs=0.1)
        assert metrics["r2"] > 0.8

    def test_crop_has_no_effect_when_data_has_none(self):
        X_train, X_test, y_train, _ = split(synthetic_frame(4000))
        pipe = build_pipeline("ridge").fit(X_train, y_train)
        spread = crop_effect(pipe, X_test, n=300)
        assert list(spread.columns) == config.CROPS
        assert (spread.max(axis=1) - spread.min(axis=1)).mean() < 0.1


class TestSelectModel:
    def test_prefers_simple_model_within_tolerance(self):
        assert select_model({"ridge": 0.5005, "hist-gradient-boosting": 0.5000}) == "ridge"

    def test_picks_best_model_when_gap_exceeds_tolerance(self):
        assert select_model({"ridge": 0.60, "hist-gradient-boosting": 0.50}) == "hist-gradient-boosting"