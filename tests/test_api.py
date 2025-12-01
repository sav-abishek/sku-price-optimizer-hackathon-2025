from fastapi.testclient import TestClient
import pytest

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.slow
def test_optimize_endpoint_returns_payload():
    payload = {
        "pinc_target": 0.0,
        "price_floor_delta": -300.0,
        "price_ceiling_delta": 500.0,
        "price_step": 50.0,
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()

    for key in ("metadata", "portfolio", "summary", "architecture", "unmapped"):
        assert key in body

    metadata = body["metadata"]
    assert isinstance(metadata.get("success"), bool)
    # Baseline run should revert to baseline but remain successful.
    assert "pinc_actual" in metadata

    assert isinstance(body["portfolio"], list)
    assert isinstance(body["summary"], list)
