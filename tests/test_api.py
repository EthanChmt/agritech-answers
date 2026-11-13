import pytest

from api.main import build_frame
from api.schemas import CROPS, ContextRequest
from tests.conftest import TEST_RMSE, VALID_CONTEXT


def predict_payload(**overrides):
    return {**VALID_CONTEXT, "crop": "Wheat", **overrides}


class TestHealth:
    def test_health_reports_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredict:
    def test_returns_prediction_with_uncertainty_band(self, client):
        response = client.post("/predict", json=predict_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["crop"] == "Wheat"
        assert body["rmse"] == TEST_RMSE
        assert body["lower_bound"] <= body["predicted_yield"] <= body["upper_bound"]
        assert body["upper_bound"] - body["predicted_yield"] == pytest.approx(TEST_RMSE, abs=1e-3)

    def test_fertilizer_and_irrigation_raise_the_yield(self, client):
        without = client.post("/predict", json=predict_payload(fertilizer_used=False, irrigation_used=False))
        with_inputs = client.post("/predict", json=predict_payload())
        gain = with_inputs.json()["predicted_yield"] - without.json()["predicted_yield"]
        assert gain == pytest.approx(2.7, abs=0.2)

    def test_more_rainfall_raises_the_yield(self, client):
        dry = client.post("/predict", json=predict_payload(rainfall_mm=200)).json()["predicted_yield"]
        wet = client.post("/predict", json=predict_payload(rainfall_mm=800)).json()["predicted_yield"]
        assert wet - dry == pytest.approx(3.0, abs=0.3)

    def test_optional_fields_have_defaults(self, client):
        response = client.post("/predict", json=predict_payload())
        assert response.status_code == 200

    def test_lower_bound_is_never_negative(self, client):
        response = client.post("/predict", json=predict_payload(
            rainfall_mm=100, temperature_celsius=15, fertilizer_used=False, irrigation_used=False))
        assert response.json()["lower_bound"] >= 0

    def test_prediction_error_returns_400(self, failing_client):
        response = failing_client.post("/predict", json=predict_payload())
        assert response.status_code == 400
        assert "Erreur lors de la prédiction" in response.json()["detail"]


INVALID_PREDICT_PAYLOADS = {
    "culture inconnue": predict_payload(crop="Banane"),
    "culture absente": {k: v for k, v in predict_payload().items() if k != "crop"},
    "pluie trop forte": predict_payload(rainfall_mm=5000),
    "pluie trop faible": predict_payload(rainfall_mm=10),
    "température trop haute": predict_payload(temperature_celsius=80),
    "température absente": {k: v for k, v in predict_payload().items() if k != "temperature_celsius"},
    "pluie non numérique": predict_payload(rainfall_mm="beaucoup"),
    "région inconnue": predict_payload(region="Nord-Ouest"),
    "jours avant récolte hors bornes": predict_payload(days_to_harvest=500),
    "ancien schéma (pesticides)": {"crop": "Wheat", "rainfall": 600, "temperature": 25, "pesticides": 1.0},
}


class TestPredictValidation:
    @pytest.mark.parametrize("payload", INVALID_PREDICT_PAYLOADS.values(), ids=INVALID_PREDICT_PAYLOADS.keys())
    def test_invalid_payload_returns_422(self, client, payload):
        assert client.post("/predict", json=payload).status_code == 422


class TestRecommend:
    def test_returns_the_six_crops_sorted_by_decreasing_yield(self, client):
        response = client.post("/recommend", json=VALID_CONTEXT)
        assert response.status_code == 200
        ranking = response.json()["ranking"]
        assert sorted(item["crop"] for item in ranking) == sorted(CROPS)
        yields = [item["predicted_yield"] for item in ranking]
        assert yields == sorted(yields, reverse=True)
        assert [item["rank"] for item in ranking] == [1, 2, 3, 4, 5, 6]

    def test_spread_is_the_gap_between_best_and_worst(self, client):
        body = client.post("/recommend", json=VALID_CONTEXT).json()
        gap = body["ranking"][0]["predicted_yield"] - body["ranking"][-1]["predicted_yield"]
        assert body["spread"] == pytest.approx(gap, abs=1e-3)

    def test_warns_when_spread_is_below_model_error(self, client):
        body = client.post("/recommend", json=VALID_CONTEXT).json()
        assert body["spread"] < body["rmse"]
        assert body["uncertainty_warning"] is True
        assert "ne permet pas de désigner une culture optimale" in body["message"]

    def test_no_warning_when_spread_exceeds_model_error(self, stub_client):
        body = stub_client.post("/recommend", json=VALID_CONTEXT).json()
        assert body["uncertainty_warning"] is False
        assert [item["crop"] for item in body["ranking"]] == list(reversed(CROPS))

    def test_each_crop_has_an_uncertainty_band(self, client):
        for item in client.post("/recommend", json=VALID_CONTEXT).json()["ranking"]:
            assert item["lower_bound"] <= item["predicted_yield"] <= item["upper_bound"]

    def test_prediction_error_returns_400(self, failing_client):
        assert failing_client.post("/recommend", json=VALID_CONTEXT).status_code == 400


INVALID_RECOMMEND_PAYLOADS = {
    "corps vide": {},
    "pluie seule": {"rainfall_mm": 600},
    "pluie hors bornes": {**VALID_CONTEXT, "rainfall_mm": 5000},
    "engrais non booléen": {**VALID_CONTEXT, "fertilizer_used": "peut-être"},
}


class TestRecommendValidation:
    @pytest.mark.parametrize("payload", INVALID_RECOMMEND_PAYLOADS.values(), ids=INVALID_RECOMMEND_PAYLOADS.keys())
    def test_invalid_payload_returns_422(self, client, payload):
        assert client.post("/recommend", json=payload).status_code == 422


class TestBuildFrame:
    def test_one_row_per_crop_with_identical_context(self):
        frame = build_frame(ContextRequest(**VALID_CONTEXT), CROPS)
        assert list(frame["Crop"]) == CROPS
        assert frame["Rainfall_mm"].nunique() == 1

    def test_booleans_are_converted_to_integers(self):
        frame = build_frame(ContextRequest(**VALID_CONTEXT), ["Wheat"])
        assert frame.loc[0, "Fertilizer_Used"] == 1
        assert frame.loc[0, "Irrigation_Used"] == 1