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
    response = client.post(
        "/api/v1/map",
        json={"mode": mode, "categories": ["Earnings", "Deductions", "Taxes"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert_category_response_shape(data)
    assert sum(len(items) for items in data.values()) == dataset_counts[0]


def test_map_endpoint_accepts_lowercase_mode(client: TestClient):
    response = client.post(
        "/api/v1/map",
        json={"mode": "max_occurrence", "categories": ["Earnings"]},
    )
    assert response.status_code == 200


def test_single_map_endpoint_returns_known_prior_code_in_category(client: TestClient):
    response = client.get(
        "/api/v1/map/ADVANCE_RECOVERY"
        "?selectedCategories=Deductions&mode=MAX_OCCURRENCE"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Earnings"] == []
    assert data["Taxes"] == []
    assert data["Deductions"][0]["priorCode"] == "ADVANCE_RECOVERY"
    assert data["Deductions"][0]["globalCode"] != "NO_MATCH"


def test_single_map_endpoint_returns_no_match_for_missing_prior_without_gpt(client: TestClient):
    response = client.get(
        "/api/v1/map/UNMAPPED_TEST_CODE_X"
        "?selectedCategories=Earnings&selectedCategories=Taxes&mode=MAX_OCCURRENCE"
    )
    assert response.status_code == 200
    data = response.json()
    assert_category_response_shape(data)
    assert data["Deductions"] == []
    for category in ["Earnings", "Taxes"]:
        assert data[category] == [
            {
                "priorCode": "UNMAPPED_TEST_CODE_X",
                "globalCode": "NO_MATCH",
            }
        ]


def test_batch_lookup_endpoint_combines_known_and_missing_prior_codes(client: TestClient):
    response = client.post(
        "/api/v1/map/batch",
        json={
            "mode": "MAX_OCCURRENCE",
            "categories": ["Deductions"],
            "priorCodes": ["ADVANCE_RECOVERY", "UNMAPPED_TEST_CODE_X"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Earnings"] == []
    assert data["Taxes"] == []
    assert data["Deductions"][0]["priorCode"] == "ADVANCE_RECOVERY"
    assert data["Deductions"][0]["globalCode"] != "NO_MATCH"
    assert {
        "priorCode": "UNMAPPED_TEST_CODE_X",
        "globalCode": "NO_MATCH",
    } in data["Deductions"]


def test_map_endpoint_does_not_return_reasoning(client: TestClient):
    response = client.post(
        "/api/v1/map",
        json={"mode": "MAX_OCCURRENCE", "categories": ["Earnings", "Deductions", "Taxes"]},
    )
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
    payload = {"mode": "MAX_OCCURRENCE", "categories": ["Earnings", "Deductions", "Taxes"]}
    first = client.post("/api/v1/map", json=payload).json()
    second = client.post("/api/v1/map", json=payload).json()
    assert first == second


def test_map_all_scopes_to_selected_category(client: TestClient):
    results = client.post(
        "/api/v1/map",
        json={"mode": "MAX_OCCURRENCE", "categories": ["Deductions"]},
    ).json()
    assert results["Earnings"] == []
    assert results["Taxes"] == []
    mapping = {item["priorCode"]: item["globalCode"] for item in results["Deductions"]}
    assert mapping["ADVANCE_RECOVERY"] != "NO_MATCH"


def test_map_all_supports_multiple_selected_categories(client: TestClient):
    results = client.post(
        "/api/v1/map",
        json={"mode": "LAST_MODIFIED_DATE", "categories": ["Earnings", "Taxes"]},
    ).json()
    assert results["Deductions"] == []
    mapping = {item["priorCode"]: item["globalCode"] for item in results["Earnings"]}
    assert mapping
    assert results["Taxes"]


def test_invalid_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={"mode": "INVALID_MODE", "categories": ["Earnings"]})
    assert response.status_code == 422


def test_missing_mode_returns_422(client: TestClient):
    response = client.post("/api/v1/map", json={"categories": ["Earnings"]})
    assert response.status_code == 422


def test_batch_lookup_requires_prior_codes(client: TestClient):
    response = client.post(
        "/api/v1/map/batch",
        json={"mode": "MAX_OCCURRENCE", "categories": ["Earnings"], "priorCodes": []},
    )
    assert response.status_code == 422


def test_category_selection_is_required(client: TestClient):
    single = client.get("/api/v1/map/ADVANCE_RECOVERY?mode=MAX_OCCURRENCE")
    batch = client.post(
        "/api/v1/map/batch",
        json={"mode": "MAX_OCCURRENCE", "priorCodes": ["ADVANCE_RECOVERY"]},
    )
    map_all = client.post("/api/v1/map", json={"mode": "MAX_OCCURRENCE"})

    assert single.status_code == 422
    assert batch.status_code == 422
    assert map_all.status_code == 422


def test_invalid_category_returns_422(client: TestClient):
    single = client.get(
        "/api/v1/map/ADVANCE_RECOVERY"
        "?selectedCategories=Benefits&mode=MAX_OCCURRENCE"
    )
    batch = client.post(
        "/api/v1/map/batch",
        json={
            "mode": "MAX_OCCURRENCE",
            "categories": ["Benefits"],
            "priorCodes": ["ADVANCE_RECOVERY"],
        },
    )

    assert single.status_code == 422
    assert batch.status_code == 422


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


def test_openapi_exposes_category_enum_array_for_swagger_selection(client: TestClient):
    schema = client.get("/openapi.json").json()
    category_schema = schema["components"]["schemas"]["PayrollCategory"]
    assert category_schema["enum"] == ["Earnings", "Deductions", "Taxes"]

    get_params = schema["paths"]["/api/v1/map/{prior_code}"]["get"]["parameters"]
    selected = next(param for param in get_params if param["name"] == "selectedCategories")
    assert selected["required"] is True
    assert selected["schema"]["type"] == "array"
    assert selected["schema"]["items"]["$ref"] == "#/components/schemas/PayrollCategory"
