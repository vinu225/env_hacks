"""
Integration Tests for Core API Endpoints.
"""

from fastapi.testclient import TestClient


def test_list_locations(client: TestClient):
    """Verify listing seeded locations."""
    response = client.get("/api/v1/locations")
    assert response.status_code == 200
    locations = response.json()
    assert len(locations) >= 4
    delhi = next((loc for loc in locations if "delhi" in loc["id"]), None)
    assert delhi is not None
    assert delhi["state"] == "Delhi"


def test_get_location_by_id(client: TestClient):
    """Verify retrieving a single location by ID."""
    response = client.get("/api/v1/locations/loc_mumbai_kurla")
    assert response.status_code == 200
    loc = response.json()
    assert loc["name"] == "Kurla (Mithi River Basin)"


def test_get_location_not_found(client: TestClient):
    """Verify 404 response for nonexistent location."""
    response = client.get("/api/v1/locations/non_existent_loc")
    assert response.status_code == 404


def test_get_heat_risk(client: TestClient):
    """Verify heat risk assessment pipeline."""
    response = client.get("/api/v1/locations/loc_delhi_central/heat")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["category"] == "HEAT"
    assert data["prediction"]["model_name"] == "rule-baseline-heat"
    assert "heat_index_c" in data
    assert "advisory_recommendation" in data


def test_get_water_risk(client: TestClient):
    """Verify water stress risk assessment pipeline."""
    response = client.get("/api/v1/locations/loc_chennai_velachery/water")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["category"] == "WATER_STRESS"
    assert "water_stress_index" in data


def test_get_composite_risk_fusion(client: TestClient):
    """Verify multi-hazard risk fusion endpoint."""
    response = client.get("/api/v1/locations/loc_delhi_central/risk")
    assert response.status_code == 200
    data = response.json()
    assert "overall_risk_score" in data
    assert "overall_risk_level" in data
    assert "sub_risks" in data
    assert "heat" in data["sub_risks"]
    assert "water" in data["sub_risks"]
    assert "rainfall" in data["sub_risks"]
    assert "primary_hazard" in data


def test_community_reports_flow(client: TestClient):
    """Verify creating and listing community incident reports."""
    payload = {
        "location_id": "loc_mumbai_kurla",
        "incident_type": "WATERLOGGING",
        "severity": "HIGH",
        "description": "Severe water accumulation under Kurla railway station bridge.",
    }
    create_resp = client.post("/api/v1/reports", json=payload)
    assert create_resp.status_code == 201
    created_report = create_resp.json()
    assert created_report["id"].startswith("rep_")
    assert created_report["verification_status"] == "UNVERIFIED"

    # Test upvote
    report_id = created_report["id"]
    upvote_resp = client.post(f"/api/v1/reports/{report_id}/upvote")
    assert upvote_resp.status_code == 200
    assert upvote_resp.json()["upvote_count"] == 1


def test_active_alerts(client: TestClient):
    """Verify retrieving active environmental alerts."""
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) >= 1
    assert "title" in alerts[0]
    assert "severity" in alerts[0]


def test_assistant_explain(client: TestClient):
    """Verify AI assistant plain-language explanation endpoint."""
    payload = {
        "location_id": "loc_delhi_central",
        "audience": "general_public",
        "language": "en",
    }
    response = client.post("/api/v1/assistant/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "actionable_guidance" in data
    assert len(data["key_bullet_points"]) > 0
