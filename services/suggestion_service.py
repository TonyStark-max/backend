from typing import Any


def generate_suggestions(
    application: dict[str, Any],
) -> list[str]:
    """
    Generate rule-based suggestions from the
    7 applicant features.

    This service provides general financial guidance.
    It does not claim that a specific feature caused
    the ML prediction.
    """

    suggestions: list[str] = []

    credit_score = application["credit_score"]
    annual_income = application["annual_income"]
    loan_amount = application["loan_amount"]
    loan_tenure = application["loan_tenure"]
    employment_type = application["employment_type"]
    dependents = application["dependents"]

    # ========================================================
    # 1. Credit score
    # ========================================================

    if credit_score < 600:
        suggestions.append(
            "Consider improving your credit score before applying."
        )

    elif credit_score < 700:
        suggestions.append(
            "A higher credit score may strengthen your loan application."
        )

    # ========================================================
    # 2. Loan amount vs annual income
    # ========================================================

    loan_to_income = loan_amount / annual_income

    if loan_to_income > 5:
        suggestions.append(
            "Consider reducing the requested loan amount "
            "or increasing your demonstrated annual income."
        )

    elif loan_to_income > 3:
        suggestions.append(
            "The requested loan amount is relatively high "
            "compared with the annual income provided."
        )

    # ========================================================
    # 3. Loan tenure
    # ========================================================

    if loan_to_income > 3 and loan_tenure < 7:
        suggestions.append(
            "A longer repayment tenure may help reduce "
            "repayment pressure."
        )

    # ========================================================
    # 4. Employment
    # ========================================================

    if employment_type == "Unemployed":
        suggestions.append(
            "A stable employment status or verified source "
            "of income may strengthen the application."
        )

    # ========================================================
    # 5. Dependents
    # ========================================================

    if dependents >= 3:
        suggestions.append(
            "Ensure that your income comfortably supports "
            "the requested loan alongside existing "
            "household responsibilities."
        )

    # ========================================================
    # 6. Positive feedback
    # ========================================================

    if not suggestions:
        suggestions.append(
            "Your provided financial profile does not indicate "
            "any major improvement areas based on the current "
            "guidance rules."
        )

    return suggestions