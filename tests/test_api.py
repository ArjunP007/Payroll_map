from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def dataset_counts() -> tuple[int, int, dict[str, int]]:
    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    category_prior_counts = {category: len(bucket) for category, bucket in raw.items()}
    return (
        sum(category_prior_counts.values()),
        sum(len(rows) for bucket in raw.values() for rows in bucket.values()),
        category_prior_counts,
    )


def assert_category_response_shape(data: dict):
    assert set(data.keys()) == {"Earnings", "Deductions", "Taxes"}
    for items in data.values():
        assert isinstance(items, list)
        for item in items:
            assert set(item.keys()) == {"priorCode", "globalCode"}


def test_health_returns_loaded_dataset(client: TestClient, dataset_counts: tuple[int, int, dict[str, int]]):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["datasetLoaded"] is True
    assert data["priorCodeCount"] == dataset_counts[0]
    assert data["recordCount"] == dataset_counts[1]


def test_root_redirects_to_health(client: TestClient):
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert response.json()["datasetLoaded"] is True


@pytest.mark.parametrize("mode", ["ONE_TO_ONE", "MAX_OCCURRENCE", "LAST_MODIFIED_DATE"])
def test_map_endpoint_accepts_all_modes(
    client: TestClient,
    dataset_counts: tuple[int, int, dict[str, int]],
    mode: str,
):
    response = client.post("/api/v1/map", json={"mode": mode})
    assert response.status_code == 200
    data = response.json()
    assert_category_response_shape(data)
    assert sum(len(items) for items in data.values()) == dataset_counts[0]


def test_map_endpoint_accepts_lowercase_mode(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "max_occurrence"})
    assert response.status_code == 200


def test_single_map_endpoint_returns_known_prior_code_in_category(client: TestClient):
    response = client.get("/api/v1/map/ADVANCE_RECOVERY?mode=MAX_OCCURRENCE")
    assert response.status_code == 200
    assert response.json() == {
        "Earnings": [],
        "Deductions": [
            {
                "priorCode": "ADVANCE_RECOVERY",
                "globalCode": "ADV_RECOVERY",
            }
        ],
        "Taxes": [],
    }


def test_single_map_endpoint_returns_no_match_for_missing_prior_without_gpt(client: TestClient):
    response = client.get("/api/v1/map/REMOTE_HOME_STIPEND?mode=MAX_OCCURRENCE")
    assert response.status_code == 200
    data = response.json()
    assert_category_response_shape(data)
    for items in data.values():
        assert items == [
            {
                "priorCode": "REMOTE_HOME_STIPEND",
                "globalCode": "NO_MATCH",
            }
        ]


def test_batch_lookup_endpoint_combines_known_and_missing_prior_codes(client: TestClient):
    response = client.post(
        "/api/v1/map/batch",
        json={
            "mode": "MAX_OCCURRENCE",
            "priorCodes": ["ADVANCE_RECOVERY", "REMOTE_HOME_STIPEND"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Deductions"][0] == {
        "priorCode": "ADVANCE_RECOVERY",
        "globalCode": "ADV_RECOVERY",
    }
    for category in ["Earnings", "Deductions", "Taxes"]:
        assert {
            "priorCode": "REMOTE_HOME_STIPEND",
            "globalCode": "NO_MATCH",
        } in data[category]


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
    for items in response.json().values():
        for item in items:
            assert not (forbidden & set(item.keys()))


def test_map_endpoint_is_deterministic(client: TestClient):
    first = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    second = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    assert first == second


def test_known_api_winner_for_max_occurrence(client: TestClient):
    results = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"}).json()
    mapping = {item["priorCode"]: item["globalCode"] for item in results["Deductions"]}
    assert mapping["ADVANCE_RECOVERY"] == "ADV_RECOVERY"


def test_known_api_winner_for_last_modified_date(client: TestClient):
    results = client.post("/api/v1/map", json={"mode": "LAST_MODIFIED_DATE"}).json()
    mapping = {item["priorCode"]: item["globalCode"] for item in results["Earnings"]}
    assert mapping["HOUSE_ALLOWANCE"] == "INSURANCE"


def test_invalid_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "INVALID_MODE"})
    assert response.status_code == 422


def test_missing_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={})
    assert response.status_code == 422


def test_batch_lookup_requires_prior_codes(client: TestClient):
    response = client.post("/api/v1/map/batch", json={"mode": "MAX_OCCURRENCE", "priorCodes": []})
    assert response.status_code == 422


def test_reload_local_rebuilds_index(client: TestClient, dataset_counts: tuple[int, int, dict[str, int]]):
    response = client.post("/api/v1/reload", json={"source": "local"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["priorCodeCount"] == dataset_counts[0]
    assert data["recordCount"] == dataset_counts[1]


def test_invalid_reload_source_returns_422(client: TestClient):
    response = client.post("/api/v1/reload", json={"source": "ftp"})
    assert response.status_code == 422


def test_prior_codes_endpoint(client: TestClient, dataset_counts: tuple[int, int, dict[str, int]]):
    response = client.get("/api/v1/prior-codes")
    assert response.status_code == 200
    data = response.json()
    assert data["totalPriorCodes"] == dataset_counts[0]
    assert len(data["Earnings"]) == dataset_counts[2]["Earnings"]
    assert len(data["Deductions"]) == dataset_counts[2]["Deductions"]
    assert len(data["Taxes"]) == dataset_counts[2]["Taxes"]
    assert "ADVANCE_RECOVERY" in data["Deductions"]


def test_openapi_mapping_result_uses_global_code_only(client: TestClient):
    schema_text = json.dumps(client.get("/openapi.json").json())
    assert "globalCode" in schema_text
    assert ("in" + "ternal" + "Code") not in schema_text
