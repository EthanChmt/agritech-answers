import pytest
from fastapi.testclient import TestClient

from api.main import app, build_frame, load_artifacts
from api.schemas import CROPS, ContextRequest
from tests.conftest import VALID_CONTEXT

pytestmark = pytest.mark.integration


def test_saved_pipeline_is_loaded_with_its_metadata(saved_model):
    artifacts = load_artifacts()
    assert artifacts is not None
    assert artifacts.rmse == pytest.approx(0.4993)


def test_input_frame_matches_the_columns_seen_at_training(saved_model):
    artifacts = load_artifacts()
    frame = build_frame(ContextRequest(**VALID_CONTEXT), CROPS)
    assert set(frame.columns) == set(artifacts.pipeline.feature_names_in_)


def test_recommend_with_saved_pipeline_returns_six_ordered_crops(saved_model):
    client = TestClient(app)
    body = client.post("/recommend", json=VALID_CONTEXT).json()
    yields = [item["predicted_yield"] for item in body["ranking"]]
    assert len(yields) == 6
    assert yields == sorted(yields, reverse=True)
    assert body["rmse"] == pytest.approx(0.4993)


def test_predict_with_saved_pipeline(saved_model):
    client = TestClient(app)
    response = client.post("/predict", json={**VALID_CONTEXT, "crop": "Rice"})
    assert response.status_code == 200
    assert response.json()["predicted_yield"] > 0


def test_health_reports_model_loaded(saved_model):
    assert TestClient(app).get("/health").json() == {"status": "ok", "model_loaded": True}


def test_missing_model_file_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr("api.main.MODEL_PATH", tmp_path / "absent.joblib")
    load_artifacts.cache_clear()
    try:
        client = TestClient(app)
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/recommend", json=VALID_CONTEXT).status_code == 503
        assert client.post("/predict", json={**VALID_CONTEXT, "crop": "Wheat"}).status_code == 503
    finally:
        load_artifacts.cache_clear()