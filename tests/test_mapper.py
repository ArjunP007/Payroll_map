from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pytest

from app.config import PrecedenceMode, TieBreakStrategy, settings
from app.index_builder import build_index
from app.loader import _normalize
from app.mapper import MODE_RESOLVERS, map_all, map_one, supported_modes
from app.schemas import NormalizedRecord


def make_record(
    prior_code: str,
    internal_code: str,
    date_str: str,
    candidate_index: int = 0,
) -> NormalizedRecord:
    return NormalizedRecord(
        priorCode=prior_code,
        internalCode=internal_code,
        lastModifiedDate=datetime.strptime(date_str, "%Y-%m-%d"),
        candidateIndex=candidate_index,
        globalIndex=candidate_index,
    )


def test_one_to_one_clean_case():
    index = build_index(
        [
            make_record("LEAVE_ENCASHMENT", "LEAVE_ENCASH", "2024-01-01", 0),
            make_record("LEAVE_ENCASHMENT", "LEAVE_ENCASH", "2023-01-01", 1),
        ]
    )
    result = map_one(index, "LEAVE_ENCASHMENT", PrecedenceMode.ONE_TO_ONE)
    assert result.internalCode == "LEAVE_ENCASH"


def test_precedence_modes_are_registered_for_dynamic_dispatch():
    assert set(supported_modes()) == set(PrecedenceMode)
    assert set(MODE_RESOLVERS) == set(PrecedenceMode)


def test_one_to_one_falls_back_for_ambiguous_codes():
    index = build_index(
        [
            make_record("BASIC_SALARY", "BASIC", "2024-01-01", 0),
            make_record("BASIC_SALARY", "BASIC", "2023-01-01", 1),
            make_record("BASIC_SALARY", "BASE", "2025-01-01", 2),
        ]
    )
    result = map_one(index, "BASIC_SALARY", PrecedenceMode.ONE_TO_ONE)
    assert result.internalCode == "BASIC"


def test_max_occurrence_clear_winner():
    index = build_index(
        [
            make_record("OVERTIME_PAY", "OT", "2022-01-01", 0),
            make_record("OVERTIME_PAY", "OT", "2022-02-01", 1),
            make_record("OVERTIME_PAY", "OVERTIME", "2024-01-01", 2),
        ]
    )
    result = map_one(index, "OVERTIME_PAY", PrecedenceMode.MAX_OCCURRENCE)
    assert result.internalCode == "OT"


def test_max_occurrence_uses_latest_date_before_final_tie_break():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2024-01-01", 0),
            make_record("PC", "BETA", "2024-06-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert result.internalCode == "BETA"


def test_tie_break_can_be_lexicographic(monkeypatch):
    monkeypatch.setattr(settings, "tie_break_strategy", TieBreakStrategy.LEXICOGRAPHIC)
    index = build_index(
        [
            make_record("PC", "ZETA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2024-01-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert result.internalCode == "ALPHA"


def test_tie_break_defaults_to_first_seen(monkeypatch):
    monkeypatch.setattr(settings, "tie_break_strategy", TieBreakStrategy.FIRST_SEEN)
    index = build_index(
        [
            make_record("PC", "ZETA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2024-01-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert result.internalCode == "ZETA"


def test_last_modified_date_clear_winner():
    index = build_index(
        [
            make_record("HOUSE_ALLOWANCE", "BASE", "2022-08-14", 0),
            make_record("HOUSE_ALLOWANCE", "HOUSE_RENT_ALLOWANCE", "2023-07-14", 1),
            make_record("HOUSE_ALLOWANCE", "INSURANCE", "2024-05-28", 2),
        ]
    )
    result = map_one(index, "HOUSE_ALLOWANCE", PrecedenceMode.LAST_MODIFIED_DATE)
    assert result.internalCode == "INSURANCE"


def test_last_modified_date_uses_latest_per_internal_code():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2022-01-01", 0),
            make_record("PC", "ALPHA", "2024-06-01", 1),
            make_record("PC", "BETA", "2024-05-31", 2),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.LAST_MODIFIED_DATE)
    assert result.internalCode == "ALPHA"


def test_last_modified_date_uses_count_before_final_tie_break():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2023-01-01", 1),
            make_record("PC", "BETA", "2024-01-01", 2),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.LAST_MODIFIED_DATE)
    assert result.internalCode == "ALPHA"


def test_map_all_returns_every_prior_code_in_order():
    index = build_index(
        [
            make_record("FIRST", "INT_A", "2024-01-01", 0),
            make_record("SECOND", "INT_B", "2024-01-01", 0),
            make_record("THIRD", "INT_C", "2024-01-01", 0),
        ]
    )
    results = map_all(index, PrecedenceMode.ONE_TO_ONE)
    assert [result.priorCode for result in results] == ["FIRST", "SECOND", "THIRD"]


def test_map_one_normalizes_lookup_code():
    index = build_index([make_record("BASIC_SALARY", "BASIC", "2024-01-01")])
    result = map_one(index, "  basic_salary  ", PrecedenceMode.ONE_TO_ONE)
    assert result.priorCode == "BASIC_SALARY"
    assert result.internalCode == "BASIC"


def test_unknown_prior_code_raises():
    index = build_index([make_record("KNOWN", "INT", "2024-01-01")])
    with pytest.raises(KeyError, match="UNKNOWN"):
        map_one(index, "UNKNOWN", PrecedenceMode.MAX_OCCURRENCE)


@pytest.fixture(scope="module")
def full_index():
    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return build_index(_normalize(raw))


def test_full_dataset_size(full_index):
    assert len(full_index.prior_codes) == 50
    assert full_index.total_records == 250


@pytest.mark.parametrize(
    ("prior_code", "expected"),
    [
        ("DEARNESS_ALLOWANCE", "TRAVEL_ALLOWANCE"),
        ("RETRO_PAYMENT", "OVERTIME"),
        ("ESI_DEDUCTION", "VACATION_PAY"),
        ("ARREAR_PAYMENT", "REGULAR_HOURS"),
        ("PROFESSIONAL_TAX", "DEARNESS"),
    ],
)
def test_full_dataset_one_to_one_known_winners(full_index, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.ONE_TO_ONE)
    assert result.internalCode == expected


@pytest.mark.parametrize(
    ("prior_code", "expected"),
    [
        ("PF_DEDUCTION", "ADVANCE"),
        ("ADVANCE_RECOVERY", "ADV_RECOVERY"),
        ("MEDICAL_ALLOWANCE", "MEDICAL"),
        ("SHIFT_ALLOWANCE", "INS_PREMIUM"),
        ("LOAN_RECOVERY", "MEAL_ALLOWANCE"),
    ],
)
def test_full_dataset_max_occurrence_known_winners(full_index, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.MAX_OCCURRENCE)
    assert result.internalCode == expected


@pytest.mark.parametrize(
    ("prior_code", "expected"),
    [
        ("HOUSE_ALLOWANCE", "INSURANCE"),
        ("ADVANCE_RECOVERY", "ADV_RECOVERY"),
        ("SHIFT_ALLOWANCE", "SPECIAL_PAY"),
        ("MEDICAL_ALLOWANCE", "HRA"),
        ("DEARNESS_ALLOWANCE", "TRAVEL_ALLOWANCE"),
    ],
)
def test_full_dataset_last_modified_known_winners(full_index, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.LAST_MODIFIED_DATE)
    assert result.internalCode == expected


def test_full_dataset_map_all_covers_every_prior_code(full_index):
    for mode in PrecedenceMode:
        results = map_all(full_index, mode)
        assert len(results) == len(full_index.prior_codes)
        assert [result.priorCode for result in results] == list(full_index.prior_codes)
        assert all(result.internalCode for result in results)
