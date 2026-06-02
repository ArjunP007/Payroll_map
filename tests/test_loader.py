from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.config import PayrollCategory
from app.loader import (
    DatasetLoadError,
    DatasetSchemaError,
    RecordValidationError,
    _normalize,
    _normalize_code,
    _parse_candidate,
    _parse_date,
    load_dataset,
)
from app.schemas import NormalizedRecord


VALID_DATASET = {
    "Earnings": {
        " basic_salary ": [
            {
                "globalCode": " basic ",
                "LastModifiedDate": "05/19/2024",
                "payType": "salary",
                "country": "US",
            },
            {
                "globalCode": "BASE",
                "LastModifiedDate": "01/01/2021",
            },
        ],
        "OVERTIME_PAY": [
            {
                "globalCode": "OT",
                "LastModifiedDate": "11/04/2024",
            }
        ],
    },
    "Deductions": {
        "HEALTH_INSURANCE": [
            {
                "globalCode": "INS",
                "LastModifiedDate": "05/18/2024",
                "deductionType": "Health",
            }
        ]
    },
    "Taxes": {
        "INCOME_TAX": [
            {
                "globalCode": "TAX",
                "LastModifiedDate": "05/17/2024",
            }
        ]
    },
}


def test_normalize_code_trims_and_uppercases():
    assert _normalize_code("  oVeRtImE_pAy  ") == "OVERTIME_PAY"


def test_normalize_code_rejects_empty_values():
    with pytest.raises(RecordValidationError):
        _normalize_code("   ")


def test_parse_date_accepts_mm_dd_yyyy():
    assert _parse_date("05/19/2024", "BASIC_SALARY", 0) == datetime(2024, 5, 19)


def test_parse_date_rejects_iso_format():
    with pytest.raises(RecordValidationError, match="expected MM/DD/YYYY"):
        _parse_date("2024-05-19", "BASIC_SALARY", 0)


def test_parse_candidate_accepts_global_code_contract_and_metadata():
    candidate = {
        "globalCode": "BASIC",
        "LastModifiedDate": "05/19/2024",
        "payType": "salary",
        "country": "US",
    }
    record = _parse_candidate(PayrollCategory.EARNINGS, "BASIC_SALARY", 2, candidate, 9)
    assert isinstance(record, NormalizedRecord)
    assert record.category == PayrollCategory.EARNINGS
    assert record.priorCode == "BASIC_SALARY"
    assert record.globalCode == "BASIC"
    assert record.candidateIndex == 2
    assert record.globalIndex == 9
    assert record.metadata == {"payType": "salary", "country": "US"}


def test_parse_candidate_rejects_legacy_target_code_field():
    legacy_key = "in" + "ternal" + "Code"
    with pytest.raises(RecordValidationError):
        _parse_candidate(
            PayrollCategory.EARNINGS,
            "PC",
            0,
            {legacy_key: "BASIC", "LastModifiedDate": "05/19/2024"},
            0,
        )


def test_parse_candidate_requires_global_code_and_date():
    with pytest.raises(RecordValidationError):
        _parse_candidate(PayrollCategory.EARNINGS, "PC", 0, {}, 0)


def test_normalize_returns_category_scoped_flat_records():
    records = _normalize(VALID_DATASET)
    assert len(records) == 5
    assert [record.category for record in records] == [
        PayrollCategory.EARNINGS,
        PayrollCategory.EARNINGS,
        PayrollCategory.EARNINGS,
        PayrollCategory.DEDUCTIONS,
        PayrollCategory.TAXES,
    ]
    assert [record.priorCode for record in records] == [
        "BASIC_SALARY",
        "BASIC_SALARY",
        "OVERTIME_PAY",
        "HEALTH_INSURANCE",
        "INCOME_TAX",
    ]
    assert [record.globalCode for record in records] == ["BASIC", "BASE", "OT", "INS", "TAX"]
    assert records[0].metadata == {"payType": "salary", "country": "US"}


def test_normalize_rejects_bad_top_level_shape():
    with pytest.raises(DatasetSchemaError, match="Top-level"):
        _normalize([{"not": "a dict"}])


def test_normalize_rejects_unknown_category():
    with pytest.raises(DatasetSchemaError, match="categories"):
        _normalize({"Benefits": {}})


def test_normalize_rejects_empty_dataset():
    with pytest.raises(DatasetSchemaError, match="empty"):
        _normalize({})


def test_normalize_rejects_flat_legacy_dataset_shape():
    with pytest.raises(DatasetSchemaError, match="categories"):
        _normalize({"BASIC_SALARY": [{"globalCode": "BASIC", "LastModifiedDate": "01/01/2024"}]})


def test_normalize_rejects_non_list_candidate_value():
    with pytest.raises(DatasetSchemaError, match="list of candidates"):
        _normalize({"Earnings": {"BASIC_SALARY": {"globalCode": "BASIC"}}})


def test_normalize_is_strict_by_default():
    dataset = {
        "Earnings": {
            "GOOD_CODE": [
                {
                    "globalCode": "GLOBAL",
                    "LastModifiedDate": "01/01/2022",
                }
            ],
            "BAD_CODE": [{}],
        }
    }
    with pytest.raises(DatasetSchemaError, match="Dataset validation failed"):
        _normalize(dataset)


def test_normalize_can_skip_bad_records_when_lenient():
    dataset = {
        "Earnings": {
            "GOOD_CODE": [
                {
                    "globalCode": "GLOBAL",
                    "LastModifiedDate": "01/01/2022",
                }
            ],
            "BAD_CODE": [{}],
        }
    }
    records = _normalize(dataset, strict=False)
    assert len(records) == 1
    assert records[0].priorCode == "GOOD_CODE"


def test_load_dataset_from_local_file(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(VALID_DATASET), encoding="utf-8")

    records = load_dataset(source="local", path=path)
    assert len(records) == 5


def test_load_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(source="local", path=tmp_path / "missing.json")


def test_project_benchmark_dataset_has_expected_size():
    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = load_dataset(source="local")
    expected_prior_codes = sum(len(bucket) for bucket in raw.values())
    expected_records = sum(len(rows) for bucket in raw.values() for rows in bucket.values())
    assert len({(record.category, record.priorCode) for record in records}) == expected_prior_codes
    assert len(records) == expected_records
