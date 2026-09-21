import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()

from jev_client import JevClient
import app as app_module
from app import app


@pytest.fixture(autouse=True)
def ensure_test_env(monkeypatch):
    """Ensure OPENROUTER_API_KEY is present and jev_client is initialized for tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")
    if app_module.jev_client is None:
        app_module.jev_client = JevClient(api_key="mock-openrouter-key")


@pytest.fixture
def client():
    # Use TestClient with raise_server_exceptions=True
    return TestClient(app)


def test_exact_match_short_circuit(client):
    """
    Exact match short-circuit (e.g. '京NC6545' vs '京NC6545')
    Should return 100%, 0.1ms, model_used='rule', and no model should be called.
    """
    with patch("jev_client.JevClient.apredict") as mock_jev, patch("app.laya_agent") as mock_laya:
        payload = {"plate_in": "京NC6545", "plate_out": "京NC6545"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["probability_same"] == 100.0
        assert data["decision"] == "same_vehicle"
        assert data["latency_ms"] <= 1.0
        assert data["badge_color"] == "emerald"
        assert data["model_used"] == "rule"
        assert data["model_name"] == "exact_match_rule"
        assert data["analysis"]["diff_count"] == 0

        # Neither model should be invoked
        assert not mock_jev.called
        if mock_laya:
            assert not mock_laya.predict.called


def test_distinct_plate_short_circuit(client):
    """
    Distinct plate short-circuit (e.g. '沪A12345' vs '浙B67890')
    Should return mismatch, 0.1ms, model_used='rule', and no model should be called.
    """
    with patch("jev_client.JevClient.apredict") as mock_jev, patch("app.laya_agent") as mock_laya:
        payload = {"plate_in": "沪A12345", "plate_out": "浙B67890"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["probability_same"] == 0.5
        assert data["decision"] == "different_vehicles"
        assert data["badge_color"] == "red"
        assert data["latency_ms"] <= 1.0
        assert data["model_used"] == "rule"
        assert data["model_name"] == "mismatch_rule"
        assert data["analysis"]["diff_count"] >= 3

        # Neither model should be invoked
        assert not mock_jev.called
        if mock_laya:
            assert not mock_laya.predict.called


def test_request_without_model_defaults_to_jev(client):
    """
    Request without `model` field -> defaults to `jev`, calls Jev,
    returns `model_used == 'jev'`, and response contains `model_used`, `model_name`.
    """
    mock_jev_result = {
        "id": "dec-test-001",
        "model": "~typesafe/jev-latest",
        "answers": {
            "is_same_vehicle": {"type": "noul", "noul": 0.95, "confidence": 0.92},
            "plate_match_decision": {
                "type": "choice",
                "choice": "same_vehicle_ocr_mismatch",
                "probabilities": {"same_vehicle_ocr_mismatch": 0.95, "distinct_different_vehicles": 0.05}
            },
            "match_confidence_score": {"type": "score", "score": 3.0}
        }
    }

    with patch("jev_client.JevClient.apredict", new_callable=AsyncMock, return_value=mock_jev_result) as mock_apredict:
        payload = {"plate_in": "京NC6545", "plate_out": "京NC0545"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["model_used"] == "jev"
        assert "model_name" in data
        assert data["model_name"] == "~typesafe/jev-latest"
        assert data["probability_same"] == 95.0
        assert data["decision"] == "same_vehicle_ocr_mismatch"
        assert data["score"] == 3.0
        assert mock_apredict.called


def test_request_with_explicit_model_jev(client):
    """
    Request with model='jev' -> uses jev.
    """
    mock_jev_result = {
        "id": "dec-test-002",
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "is_same_vehicle": {"type": "noul", "noul": 0.88, "confidence": 0.85},
            "plate_match_decision": {
                "type": "choice",
                "choice": "same_vehicle_ocr_mismatch",
                "probabilities": {"same_vehicle_ocr_mismatch": 0.88, "distinct_different_vehicles": 0.12}
            },
            "match_confidence_score": {"type": "score", "score": 2.5}
        }
    }

    with patch("jev_client.JevClient.apredict", new_callable=AsyncMock, return_value=mock_jev_result) as mock_apredict:
        payload = {"plate_in": "京NC6545", "plate_out": "京NC0545", "model": "jev"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["model_used"] == "jev"
        assert data["model_name"] == "typesafe/jev-1.13-20260917"
        assert data["probability_same"] == 88.0
        assert mock_apredict.called


def test_request_with_model_laya(client):
    """
    Request with model='laya' -> lazy loads laya and calls laya_agent.predict in threadpool.
    """
    mock_laya_agent = MagicMock()
    mock_laya_agent.predict.return_value = {
        "answers": {
            "is_same_vehicle": {"type": "noul", "noul": 0.82, "confidence": 0.80},
            "plate_match_decision": {
                "type": "choice",
                "choice": "same_vehicle_ocr_mismatch",
                "probabilities": {"same_vehicle_ocr_mismatch": 0.82, "distinct_different_vehicles": 0.18}
            },
            "match_confidence_score": {"type": "score", "score": 2.0}
        }
    }

    with patch("laya.load", return_value=mock_laya_agent) as mock_load:
        app_module.laya_agent = None

        payload = {"plate_in": "京NC6545", "plate_out": "京NC0545", "model": "laya"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["model_used"] == "laya"
        assert data["model_name"] == "convaiinnovations/laya"
        assert data["probability_same"] == 82.0
        assert mock_load.called
        assert mock_laya_agent.predict.called


def test_model_inference_failure_raises_502_jev(client):
    """
    When Jev inference fails with exception, app should return 502 with formatted detail.
    """
    with patch("jev_client.JevClient.apredict", new_callable=AsyncMock, side_effect=RuntimeError("OpenRouter Timeout")):
        payload = {"plate_in": "京NC6545", "plate_out": "京NC0545", "model": "jev"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 502
        assert resp.json()["detail"] == "决策模型推断失败 (jev): OpenRouter Timeout"


def test_model_inference_failure_raises_502_laya(client):
    """
    When Laya inference fails with exception, app should return 502 with formatted detail.
    """
    mock_agent = MagicMock()
    mock_agent.predict.side_effect = RuntimeError("CUDA OOM")
    with patch("laya.load", return_value=mock_agent):
        app_module.laya_agent = None

        payload = {"plate_in": "京NC6545", "plate_out": "京NC0545", "model": "laya"}
        resp = client.post("/api/match", json=payload)
        assert resp.status_code == 502
        assert resp.json()["detail"] == "决策模型推断失败 (laya): CUDA OOM"
