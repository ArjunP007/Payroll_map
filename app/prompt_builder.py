"""Prompt construction for bounded GPT-assisted mapping tasks."""

from __future__ import annotations

from collections.abc import Sequence

NO_MATCH_INTERNAL_CODE = "NO_MATCH"


def build_missing_prior_prompt(prior_code: str, candidate_codes: Sequence[str]) -> str:
    """Build the constrained prompt for missing historical prior-code fallback."""

    normalized_prior_code = prior_code.strip().upper()
    catalog = "\n".join(candidate_codes)
    return (
        "You are a payroll mapping engine.\n\n"
        "Task:\n"
        "Recommend the BEST matching payroll internal code.\n\n"
        f"Input Prior Code:\n{normalized_prior_code}\n\n"
        f"Available Internal Codes:\n{catalog}\n\n"
        "Rules:\n"
        "1. Select EXACTLY ONE internal code.\n"
        "2. Use payroll semantic similarity.\n"
        "3. No explanation.\n"
        "4. No reasoning.\n"
        "5. Return ONLY the selected internal code.\n"
        f"6. If no reasonable match exists, return {NO_MATCH_INTERNAL_CODE}."
    )
