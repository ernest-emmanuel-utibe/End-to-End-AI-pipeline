import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Import components from your project files
from pipeline.pipeline import EndToEndAIPipeline
from server.main import app

# --- FIXTURES ---
@pytest.fixture
def mock_pipeline():
    """Initializes a clean instance of the pipeline for isolated unit testing."""
    return EndToEndAIPipeline()

@pytest.fixture
def api_client():
    """Provides a virtualized client interface to test the FastAPI app routing."""
    return TestClient(app)


# --- PIPELINE UNIT TESTS ---
def test_data_ingestion_shape_and_columns(mock_pipeline):
    """Verifies that the ingested streaming data matches structural criteria."""
    df = mock_pipeline.ingest_and_version_data()
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 200
    assert "user_metadata" in df.columns
    assert "target" in df.columns


def test_security_gateway_scrubs_pii(mock_pipeline):
    """Confirms that the security gateway catches and completely cleans raw PII leaks."""
    # Setup malicious leaking payload
    leaking_df = pd.DataFrame({
        "feature_1": [0.5, 0.2],
        "feature_2": [0.1, 0.8],
        "feature_3": [0.4, 0.9],
        "feature_4": [0.7, 0.3],
        "user_metadata": ["attacker_email@leakdomain.com", "clean_record"],
        "target": [1, 0]
    })
    
    clean_df = mock_pipeline.secure_and_sanitize_data(leaking_df)
    
    # Assert column removal or scrubbing verification patterns
    assert "user_metadata" not in clean_df.columns
    assert clean_df.shape[1] == 5  # 4 features + 1 target


def test_performance_gate_logic(mock_pipeline):
    """Validates that performance gates pass and fail at correct mathematical thresholds."""
    assert mock_pipeline.evaluate_performance_gate(0.85) is True
    assert mock_pipeline.evaluate_performance_gate(0.72) is False


# --- API ARCHITECTURE MOCK TESTS ---
def test_health_endpoint_degraded_state(api_client, mocker):
    """Verifies API falls back gracefully when an MLflow container isn't ready."""
    # Force the global model pointer back to None to simulate boot failures
    mocker.patch("server.main.production_model", None)
    
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_predict_endpoint_missing_payload(api_client):
    """Validates input schema structural error handling on empty data streams."""
    response = api_client.post("/predict", json={})
    assert response.status_code == 400
    assert "cannot be null or empty" in response.json()["detail"]
