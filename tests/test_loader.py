from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

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
    " basic_salary ": [
        {
            "internalCode": " basic ",
            "internalDescription": "Regular Basic",
            "LastModifiedDate": "05/19/2024",
        },
        {
            "internalCode": "BASE",
            "internalDescription": "Ignored by engine",
            "LastModifiedDate": "01/01/2021",
        },
    ],
    "OVERTIME_PAY": [
        {
            "internalCode": "OT",
            "internalDescription": "Overtime Hours",
            "LastModifiedDate": "11/04/2024",
        }
    ],
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


def test_parse_candidate_ignores_internal_description():
    candidate = {
        "internalCode": "BASIC",
        "internalDescription": "Must not affect mapping",
        "LastModifiedDate": "05/19/2024",
    }
    record = _parse_candidate("BASIC_SALARY", 2, candidate, 9)
    assert isinstance(record, NormalizedRecord)
    assert record.priorCode == "BASIC_SALARY"
    assert record.internalCode == "BASIC"
    assert record.candidateIndex == 2
    assert record.globalIndex == 9
    assert not hasattr(record, "internalDescription")


def test_parse_candidate_requires_internal_code_and_date():
    with pytest.raises(RecordValidationError):
        _parse_candidate("PC", 0, {"internalDescription": "missing"}, 0)


def test_normalize_returns_flat_records():
    records = _normalize(VALID_DATASET)
    assert len(records) == 3
    assert [record.priorCode for record in records] == [
        "BASIC_SALARY",
        "BASIC_SALARY",
        "OVERTIME_PAY",
    ]
    assert [record.internalCode for record in records] == ["BASIC", "BASE", "OT"]


def test_normalize_rejects_bad_top_level_shape():
    with pytest.raises(DatasetSchemaError, match="Top-level"):
        _normalize([{"not": "a dict"}])


def test_normalize_rejects_empty_dataset():
    with pytest.raises(DatasetSchemaError, match="empty"):
        _normalize({})


def test_normalize_rejects_non_list_candidate_value():
    with pytest.raises(DatasetSchemaError, match="list of candidates"):
        _normalize({"BASIC_SALARY": {"internalCode": "BASIC"}})


def test_normalize_is_strict_by_default():
    dataset = {
        "GOOD_CODE": [
            {
                "internalCode": "INTERNAL",
                "internalDescription": "desc",
                "LastModifiedDate": "01/01/2022",
            }
        ],
        "BAD_CODE": [{"internalDescription": "missing required fields"}],
    }
    with pytest.raises(DatasetSchemaError, match="Dataset validation failed"):
        _normalize(dataset)


def test_normalize_can_skip_bad_records_when_lenient():
    dataset = {
        "GOOD_CODE": [
            {
                "internalCode": "INTERNAL",
                "internalDescription": "desc",
                "LastModifiedDate": "01/01/2022",
            }
        ],
        "BAD_CODE": [{"internalDescription": "missing required fields"}],
    }
    records = _normalize(dataset, strict=False)
    assert len(records) == 1
    assert records[0].priorCode == "GOOD_CODE"


def test_load_dataset_from_local_file(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(VALID_DATASET), encoding="utf-8")

    records = load_dataset(source="local", path=path)
    assert len(records) == 3


def test_load_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(source="local", path=tmp_path / "missing.json")


def test_project_benchmark_dataset_has_expected_size():
    records = load_dataset(source="local")
    assert len({record.priorCode for record in records}) == 50
    assert len(records) == 250
