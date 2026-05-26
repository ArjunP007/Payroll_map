from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_loaded_dataset(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["datasetLoaded"] is True
    assert data["priorCodeCount"] == 50
    assert data["recordCount"] == 250


def test_root_redirects_to_health(client: TestClient):
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert response.json()["datasetLoaded"] is True


@pytest.mark.parametrize("mode", ["ONE_TO_ONE", "MAX_OCCURRENCE", "LAST_MODIFIED_DATE"])
def test_map_endpoint_accepts_all_modes(client: TestClient, mode: str):
    response = client.post("/api/v1/map", json={"mode": mode})
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 50


def test_map_endpoint_accepts_lowercase_mode(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "max_occurrence"})
    assert response.status_code == 200


def test_map_endpoint_returns_strict_result_array(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"})
    results = response.json()
    assert isinstance(results, list)
    for item in results:
        assert set(item.keys()) == {"priorCode", "internalCode"}


def test_map_endpoint_does_not_return_reasoning(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"})
    forbidden = {
        "candidates",
        "occurrenceCounts",
        "latestDates",
        "tieBreakApplied",
        "gptAdjudicated",
        "reason",
        "explanation",
        "confidence",
        "score",
        "mode",
        "totalMapped",
    }
    for item in response.json():
        assert not (forbidden & set(item.keys()))


def test_map_endpoint_is_deterministic(client: TestClient):
    first = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    second = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    assert first == second


def test_known_api_winner_for_max_occurrence(client: TestClient):
    results = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    mapping = {item["priorCode"]: item["internalCode"] for item in results}
    assert mapping["ADVANCE_RECOVERY"] == "ADV_RECOVERY"


def test_known_api_winner_for_last_modified_date(client: TestClient):
    results = client.post("/api/v1/map", json={"mode": "LAST_MODIFIED_DATE"}).json()
    mapping = {item["priorCode"]: item["internalCode"] for item in results}
    assert mapping["HOUSE_ALLOWANCE"] == "INSURANCE"


def test_invalid_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "INVALID_MODE"})
    assert response.status_code == 422


def test_missing_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={})
    assert response.status_code == 422


def test_reload_local_rebuilds_index(client: TestClient):
    response = client.post("/api/v1/reload", json={"source": "local"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["priorCodeCount"] == 50
    assert data["recordCount"] == 250


def test_invalid_reload_source_returns_422(client: TestClient):
    response = client.post("/api/v1/reload", json={"source": "ftp"})
    assert response.status_code == 422


def test_prior_codes_endpoint(client: TestClient):
    response = client.get("/api/v1/prior-codes")
    assert response.status_code == 200
    data = response.json()
    assert data["totalPriorCodes"] == 50
    assert "ADVANCE_RECOVERY" in data["priorCodes"]
