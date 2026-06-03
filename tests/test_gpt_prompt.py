from __future__ import annotations

import json

import pytest

from app.config import PayrollCategory
from app.exceptions import GPTAdjudicationError
from app.gpt_client import GptClient
from app.prompt_builder import NO_MATCH_GLOBAL_CODE, build_missing_prior_batch_prompt


CATALOGS = {
    PayrollCategory.EARNINGS: ("BASIC_PAY",),
    PayrollCategory.DEDUCTIONS: ("INSURANCE",),
    PayrollCategory.TAXES: ("INCOME_TAX",),
}


def test_batch_prompt_is_category_scoped_and_metadata_aware():
    prompt = build_missing_prior_batch_prompt(
        prior_codes_by_category={PayrollCategory.EARNINGS: ["remote_home_stipend"]},
        catalogs={PayrollCategory.EARNINGS: CATALOGS[PayrollCategory.EARNINGS]},
        catalog_evidence={
            PayrollCategory.EARNINGS: [
                {
                    "priorCode": "REG",
                    "globalCode": "BASIC_PAY",
                    "metadata": {"payType": "salary", "country": "US"},
                }
            ],
            PayrollCategory.DEDUCTIONS: [],
            PayrollCategory.TAXES: [],
        },
    )

    assert "Earnings" in prompt
    assert "Deductions" in prompt
    assert "Taxes" in prompt
    assert NO_MATCH_GLOBAL_CODE in prompt
    assert "candidateMetadata" in prompt
    assert "internalConfidenceGuidance" in prompt
    assert "REMOTE_HOME_STIPEND" in prompt
    assert "structured JSON only" in prompt


def test_gpt_structured_response_parser_accepts_catalog_codes_and_no_match():
    response = GptClient._parse_category_response(
        json.dumps(
            {
                "Earnings": [{"priorCode": "remote_home_stipend", "globalCode": "basic_pay"}],
                "Deductions": [{"priorCode": "unknown", "globalCode": NO_MATCH_GLOBAL_CODE}],
                "Taxes": [],
            }
        ),
        allowed_catalogs=CATALOGS,
    )

    assert response.Earnings[0].priorCode == "REMOTE_HOME_STIPEND"
    assert response.Earnings[0].globalCode == "BASIC_PAY"
    assert response.Deductions[0].globalCode == NO_MATCH_GLOBAL_CODE


def test_gpt_structured_response_parser_rejects_invented_codes():
    with pytest.raises(GPTAdjudicationError, match="not allowed"):
        GptClient._parse_category_response(
            json.dumps(
                {
                    "Earnings": [{"priorCode": "REMOTE_HOME_STIPEND", "globalCode": "MADE_UP"}],
                    "Deductions": [],
                    "Taxes": [],
                }
            ),
            allowed_catalogs=CATALOGS,
        )


def test_gpt_structured_response_parser_rejects_extra_keys():
    with pytest.raises(GPTAdjudicationError, match="exactly"):
        GptClient._parse_category_response(
            json.dumps(
                {
                    "Earnings": [
                        {
                            "priorCode": "REMOTE_HOME_STIPEND",
                            "globalCode": "BASIC_PAY",
                            "reason": "not public",
                        }
                    ],
                    "Deductions": [],
                    "Taxes": [],
                }
            ),
            allowed_catalogs=CATALOGS,
        )
