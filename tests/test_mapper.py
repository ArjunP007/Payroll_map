from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pytest

from app.config import PayrollCategory, PrecedenceMode, TieBreakStrategy, settings
from app.index_builder import build_index
from app.loader import _normalize
from app.mapper import MODE_RESOLVERS, map_all, map_batch, map_one, supported_modes
from app.prompt_builder import NO_MATCH_GLOBAL_CODE
from app.schemas import CategoryMappingResponse, MappingResult, NormalizedRecord


class FailingGptClient:
    def recommend_global_code(self, **kwargs):
        raise AssertionError("GPT should not be called")

    def recommend_global_codes(self, **kwargs):
        raise AssertionError("GPT should not be called")


class RecommendingGptClient:
    def __init__(self, response: CategoryMappingResponse) -> None:
        self.response = response
        self.received_catalogs = None
        self.received_evidence = None
        self.received_prior_codes: list[str] | None = None

    def recommend_global_code(self, **kwargs) -> CategoryMappingResponse:
        self.received_prior_codes = [kwargs["prior_code"]]
        self.received_catalogs = kwargs["catalogs"]
        self.received_evidence = kwargs["catalog_evidence"]
        return self.response

    def recommend_global_codes(self, **kwargs) -> CategoryMappingResponse:
        self.received_prior_codes = list(kwargs["prior_codes"])
        self.received_catalogs = kwargs["catalogs"]
        self.received_evidence = kwargs["catalog_evidence"]
        return self.response


def make_record(
    prior_code: str,
    global_code: str,
    date_str: str,
    candidate_index: int = 0,
    *,
    category: PayrollCategory = PayrollCategory.EARNINGS,
    metadata: dict[str, object] | None = None,
) -> NormalizedRecord:
    return NormalizedRecord(
        category=category,
        priorCode=prior_code,
        globalCode=global_code,
        lastModifiedDate=datetime.strptime(date_str, "%Y-%m-%d"),
        candidateIndex=candidate_index,
        globalIndex=candidate_index,
        metadata=metadata or {},
    )


def first_result(response: CategoryMappingResponse, category: PayrollCategory) -> MappingResult:
    return response.as_category_map()[category][0]


def test_one_to_one_clean_case():
    index = build_index(
        [
            make_record("LEAVE_ENCASHMENT", "LEAVE_ENCASH", "2024-01-01", 0),
            make_record("LEAVE_ENCASHMENT", "LEAVE_ENCASH", "2023-01-01", 1),
        ]
    )
    result = map_one(index, "LEAVE_ENCASHMENT", PrecedenceMode.ONE_TO_ONE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "LEAVE_ENCASH"


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
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "BASIC"


def test_max_occurrence_clear_winner():
    index = build_index(
        [
            make_record("OVERTIME_PAY", "OT", "2022-01-01", 0),
            make_record("OVERTIME_PAY", "OT", "2022-02-01", 1),
            make_record("OVERTIME_PAY", "OVERTIME", "2024-01-01", 2),
        ]
    )
    result = map_one(index, "OVERTIME_PAY", PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "OT"


def test_max_occurrence_uses_latest_date_before_final_tie_break():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2024-01-01", 0),
            make_record("PC", "BETA", "2024-06-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "BETA"


def test_tie_break_can_be_lexicographic(monkeypatch):
    monkeypatch.setattr(settings, "tie_break_strategy", TieBreakStrategy.LEXICOGRAPHIC)
    index = build_index(
        [
            make_record("PC", "ZETA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2024-01-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "ALPHA"


def test_tie_break_defaults_to_first_seen(monkeypatch):
    monkeypatch.setattr(settings, "tie_break_strategy", TieBreakStrategy.FIRST_SEEN)
    index = build_index(
        [
            make_record("PC", "ZETA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2024-01-01", 1),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "ZETA"


def test_last_modified_date_clear_winner():
    index = build_index(
        [
            make_record("HOUSE_ALLOWANCE", "BASE", "2022-08-14", 0),
            make_record("HOUSE_ALLOWANCE", "HOUSE_RENT_ALLOWANCE", "2023-07-14", 1),
            make_record("HOUSE_ALLOWANCE", "INSURANCE", "2024-05-28", 2),
        ]
    )
    result = map_one(index, "HOUSE_ALLOWANCE", PrecedenceMode.LAST_MODIFIED_DATE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "INSURANCE"


def test_last_modified_date_uses_latest_per_global_code():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2022-01-01", 0),
            make_record("PC", "ALPHA", "2024-06-01", 1),
            make_record("PC", "BETA", "2024-05-31", 2),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.LAST_MODIFIED_DATE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "ALPHA"


def test_last_modified_date_uses_count_before_final_tie_break():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2024-01-01", 0),
            make_record("PC", "ALPHA", "2023-01-01", 1),
            make_record("PC", "BETA", "2024-01-01", 2),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.LAST_MODIFIED_DATE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "ALPHA"


def test_deterministic_mapping_is_category_scoped_for_same_prior_code():
    index = build_index(
        [
            make_record("SHARED_CODE", "EARN", "2024-01-01", category=PayrollCategory.EARNINGS),
            make_record("SHARED_CODE", "DED", "2024-01-01", category=PayrollCategory.DEDUCTIONS),
        ]
    )
    result = map_one(index, "shared_code", PrecedenceMode.ONE_TO_ONE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "EARN"
    assert first_result(result, PayrollCategory.DEDUCTIONS).globalCode == "DED"
    assert result.Taxes == []


def test_extra_metadata_does_not_affect_deterministic_scoring():
    index = build_index(
        [
            make_record("PC", "ALPHA", "2024-01-01", 0, metadata={"payType": "x"}),
            make_record("PC", "BETA", "2024-06-01", 1, metadata={"payType": "better"}),
        ]
    )
    result = map_one(index, "PC", PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "BETA"


def test_map_all_returns_every_prior_code_in_category_order():
    index = build_index(
        [
            make_record("FIRST", "A", "2024-01-01", 0, category=PayrollCategory.EARNINGS),
            make_record("SECOND", "B", "2024-01-01", 0, category=PayrollCategory.DEDUCTIONS),
            make_record("THIRD", "C", "2024-01-01", 0, category=PayrollCategory.TAXES),
        ]
    )
    results = map_all(index, PrecedenceMode.ONE_TO_ONE)
    assert [item.priorCode for item in results.Earnings] == ["FIRST"]
    assert [item.priorCode for item in results.Deductions] == ["SECOND"]
    assert [item.priorCode for item in results.Taxes] == ["THIRD"]


def test_known_prior_code_does_not_use_missing_prior_gpt_fallback():
    index = build_index([make_record("KNOWN", "GLOBAL", "2024-01-01")])
    result = map_one(
        index,
        "KNOWN",
        PrecedenceMode.MAX_OCCURRENCE,
        gpt_client=FailingGptClient(),
    )
    assert first_result(result, PayrollCategory.EARNINGS).globalCode == "GLOBAL"


def test_missing_prior_code_uses_gpt_recommendation_from_category_catalogs():
    index = build_index(
        [
            make_record("REG", "BASIC_PAY", "2024-01-01", metadata={"payType": "regular"}),
            make_record("TAX", "INCOME_TAX", "2024-01-01", category=PayrollCategory.TAXES),
        ]
    )
    gpt_response = CategoryMappingResponse(
        Earnings=[MappingResult(priorCode="REMOTE_HOME_STIPEND", globalCode="BASIC_PAY")],
        Deductions=[],
        Taxes=[],
    )
    gpt_client = RecommendingGptClient(gpt_response)

    result = map_one(
        index,
        "REMOTE_HOME_STIPEND",
        PrecedenceMode.MAX_OCCURRENCE,
        gpt_client=gpt_client,
    )

    assert result.model_dump() == {
        "Earnings": [{"priorCode": "REMOTE_HOME_STIPEND", "globalCode": "BASIC_PAY"}],
        "Deductions": [],
        "Taxes": [],
    }
    assert gpt_client.received_prior_codes == ["REMOTE_HOME_STIPEND"]
    assert gpt_client.received_catalogs[PayrollCategory.EARNINGS] == ("BASIC_PAY",)
    assert gpt_client.received_evidence[PayrollCategory.EARNINGS][0]["metadata"] == {
        "payType": "regular"
    }


def test_missing_prior_code_returns_no_match_without_gpt_client():
    index = build_index([make_record("KNOWN", "GLOBAL", "2024-01-01")])
    result = map_one(index, "UNKNOWN", PrecedenceMode.MAX_OCCURRENCE)
    for category_results in result.as_category_map().values():
        assert category_results == [
            MappingResult(priorCode="UNKNOWN", globalCode=NO_MATCH_GLOBAL_CODE)
        ]


def test_batch_lookup_combines_deterministic_and_gpt_results():
    index = build_index(
        [
            make_record("KNOWN", "GLOBAL", "2024-01-01"),
            make_record("TAXABLE", "TAX", "2024-01-01", category=PayrollCategory.TAXES),
        ]
    )
    gpt_response = CategoryMappingResponse(
        Earnings=[],
        Deductions=[],
        Taxes=[MappingResult(priorCode="UNKNOWN_TAX", globalCode="TAX")],
    )
    gpt_client = RecommendingGptClient(gpt_response)

    result = map_batch(
        index,
        ["KNOWN", "UNKNOWN_TAX"],
        PrecedenceMode.ONE_TO_ONE,
        gpt_client=gpt_client,
    )

    assert result.Earnings == [MappingResult(priorCode="KNOWN", globalCode="GLOBAL")]
    assert result.Taxes == [MappingResult(priorCode="UNKNOWN_TAX", globalCode="TAX")]
    assert gpt_client.received_prior_codes == ["UNKNOWN_TAX"]


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
    assert full_index.total_prior_codes > 0
    assert full_index.total_records == sum(
        category_index.candidate_count(prior_code)
        for category_index in full_index.category_indexes.values()
        for prior_code in category_index.prior_codes
    )


@pytest.mark.parametrize(
    ("category", "prior_code", "expected"),
    [
        (PayrollCategory.EARNINGS, "DEARNESS_ALLOWANCE", "TRAVEL_ALLOWANCE"),
        (PayrollCategory.EARNINGS, "RETRO_PAYMENT", "OVERTIME"),
        (PayrollCategory.DEDUCTIONS, "ESI_DEDUCTION", "VACATION_PAY"),
        (PayrollCategory.EARNINGS, "ARREAR_PAYMENT", "REGULAR_HOURS"),
        (PayrollCategory.TAXES, "PROFESSIONAL_TAX", "DEARNESS"),
    ],
)
def test_full_dataset_one_to_one_known_winners(full_index, category, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.ONE_TO_ONE)
    assert first_result(result, category).globalCode == expected


@pytest.mark.parametrize(
    ("category", "prior_code", "expected"),
    [
        (PayrollCategory.DEDUCTIONS, "PF_DEDUCTION", "ADVANCE"),
        (PayrollCategory.DEDUCTIONS, "ADVANCE_RECOVERY", "ADV_RECOVERY"),
        (PayrollCategory.EARNINGS, "MEDICAL_ALLOWANCE", "MEDICAL"),
        (PayrollCategory.EARNINGS, "SHIFT_ALLOWANCE", "INS_PREMIUM"),
        (PayrollCategory.DEDUCTIONS, "LOAN_RECOVERY", "MEAL_ALLOWANCE"),
    ],
)
def test_full_dataset_max_occurrence_known_winners(full_index, category, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.MAX_OCCURRENCE)
    assert first_result(result, category).globalCode == expected


@pytest.mark.parametrize(
    ("category", "prior_code", "expected"),
    [
        (PayrollCategory.EARNINGS, "HOUSE_ALLOWANCE", "INSURANCE"),
        (PayrollCategory.DEDUCTIONS, "ADVANCE_RECOVERY", "ADV_RECOVERY"),
        (PayrollCategory.EARNINGS, "SHIFT_ALLOWANCE", "SPECIAL_PAY"),
        (PayrollCategory.EARNINGS, "MEDICAL_ALLOWANCE", "HRA"),
        (PayrollCategory.EARNINGS, "DEARNESS_ALLOWANCE", "TRAVEL_ALLOWANCE"),
    ],
)
def test_full_dataset_last_modified_known_winners(full_index, category, prior_code, expected):
    result = map_one(full_index, prior_code, PrecedenceMode.LAST_MODIFIED_DATE)
    assert first_result(result, category).globalCode == expected


def test_full_dataset_map_all_covers_every_category_prior_code(full_index):
    for mode in PrecedenceMode:
        results = map_all(full_index, mode)
        for category, category_index in full_index.category_indexes.items():
            actual = [result.priorCode for result in results.as_category_map()[category]]
            assert actual == list(category_index.prior_codes)
            assert all(result.globalCode for result in results.as_category_map()[category])
