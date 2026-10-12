"""
Integration Tests for Health and Readiness Probes.
"""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Verify application liveness probe returns HTTP 200 with required metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_readiness_check(client: TestClient):
    """Verify readiness probe verifies underlying storage and provider readiness."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert data["checks"]["storage_backend"]["status"] == "healthy"
    assert data["checks"]["weather_provider"]["status"] == "healthy"


def test_root_endpoint(client: TestClient):
    """Verify root discovery endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["api_v1"] == "/api/v1"
