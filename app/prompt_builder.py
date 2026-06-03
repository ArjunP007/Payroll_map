"""Prompt construction for bounded GPT-assisted missing-prior mapping tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.config import PAYROLL_CATEGORIES, PayrollCategory

NO_MATCH_GLOBAL_CODE = "NO_MATCH"


def build_missing_prior_prompt(
    prior_code: str,
    category: PayrollCategory,
    catalogs: Mapping[PayrollCategory, Sequence[str]],
    catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, Any]]] | None = None,
) -> str:
    """Build the constrained prompt for one missing historical prior code."""

    return build_missing_prior_batch_prompt(
        prior_codes_by_category={category: [prior_code]},
        catalogs=catalogs,
        catalog_evidence=catalog_evidence,
    )


def build_missing_prior_batch_prompt(
    prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
    catalogs: Mapping[PayrollCategory, Sequence[str]],
    catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, Any]]] | None = None,
) -> str:
    """Build the constrained prompt for category-aware batch fallback."""

    selected_categories = tuple(
        category
        for category in PAYROLL_CATEGORIES
        if prior_codes_by_category.get(category)
    )
    normalized_prior_codes_by_category = {
        category.value: [
            prior_code.strip().upper()
            for prior_code in prior_codes_by_category.get(category, ())
        ]
        for category in selected_categories
    }
    payload = {
        "missingPriorCodesByCategory": normalized_prior_codes_by_category,
        "selectedCategories": [category.value for category in selected_categories],
        "categories": _category_catalog_payload(catalogs, catalog_evidence or {}),
        "internalConfidenceGuidance": {
            "purpose": "Use confidence only for internal selection. Never return confidence.",
            "minimumRecommendedConfidence": 0.72,
            "lowConfidenceAction": NO_MATCH_GLOBAL_CODE,
        },
        "requiredOutputShape": {
            category.value: [{"priorCode": "<input prior code>", "globalCode": "<catalog code or NO_MATCH>"}]
            for category in PAYROLL_CATEGORIES
        },
    }
    return (
        "You are a payroll mapping engine.\n\n"
        "Task:\n"
        "Map missing payroll prior codes to category-scoped payroll global codes.\n\n"
        "Inputs are separated by category. Treat Earnings, Deductions, and Taxes as separate "
        "namespaces. Use candidate metadata when present. If metadata is absent or sparse, use "
        "the global code name, category scope, semantic similarity, lexical similarity, and "
        "payroll meaning.\n\n"
        "Rules:\n"
        "1. Return structured JSON only.\n"
        "2. Return exactly the category keys Earnings, Deductions, and Taxes.\n"
        "3. Each item must contain exactly priorCode and globalCode.\n"
        "4. Choose globalCode only from the catalog for that same category.\n"
        f"5. Use {NO_MATCH_GLOBAL_CODE} when no confident mapping exists.\n"
        "6. Never invent global codes.\n"
        "7. Produce confidence internally for selection quality, but never return confidence.\n"
        "8. Do not include prose, markdown, scores, explanations, or extra keys.\n"
        "9. For categories without requested missing prior codes, return an empty list.\n\n"
        "Payload:\n"
        f"{json.dumps(payload, default=str, separators=(',', ':'))}"
    )


def _category_catalog_payload(
    catalogs: Mapping[PayrollCategory, Sequence[str]],
    catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        category.value: {
            "globalCodes": sorted(str(code).strip().upper() for code in catalogs.get(category, ())),
            "candidateMetadata": list(catalog_evidence.get(category, ())),
        }
        for category in PAYROLL_CATEGORIES
    }
