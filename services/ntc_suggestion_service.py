from typing import Any


def generate_ntc_suggestions(
    application: dict[str, Any],
    prediction_status: str = "Approved",
) -> list[str]:
    """
    Generate suggestions for New-To-Credit (NTC) applicants.

    These suggestions are based only on the six available
    applicant features and do not infer or estimate
    any credit score.
    """

    suggestions: list[str] = []

    annual_income = application["annual_income"]
    monthly_expenses = application.get("monthly_expenses", 0)
    monthly_income = annual_income / 12
    expense_ratio = (monthly_expenses / monthly_income) if monthly_income > 0 else 0
    disposable_income = monthly_income - monthly_expenses

    loan_amount = application["loan_amount"]
    loan_tenure = application["loan_tenure"]
    employment_type = application["employment_type"]
    dependents = application["dependents"]

    # --------------------------------------------------------
    # Expense Ratio
    # --------------------------------------------------------
    
    if disposable_income < 0:
        suggestions.append(
            "Your monthly expenses exceed your monthly income. Consider reducing monthly expenses or increasing income before applying."
        )
    elif expense_ratio > 0.6:
        suggestions.append(
            "Your monthly expenses consume a large portion of your income. Reduce monthly expenses to improve repayment capacity."
        )

    # --------------------------------------------------------
    # Loan amount vs income
    # --------------------------------------------------------

    loan_to_income = loan_amount / annual_income

    if loan_to_income > 5:
        suggestions.append(
            "Consider requesting a lower loan amount or increasing your demonstrated annual income."
        )

    elif loan_to_income > 3:
        suggestions.append(
            "Your requested loan amount is relatively high compared to your annual income."
        )

    # --------------------------------------------------------
    # Loan tenure
    # --------------------------------------------------------

    if loan_to_income > 3 and loan_tenure < 7:
        suggestions.append(
            "Choosing a longer repayment tenure may improve affordability."
        )

    # --------------------------------------------------------
    # Employment
    # --------------------------------------------------------

    if employment_type == "Unemployed":
        suggestions.append(
            "A stable source of income or employment may strengthen your application."
        )

    # --------------------------------------------------------
    # Dependents
    # --------------------------------------------------------

    if dependents >= 3:
        suggestions.append(
            "Ensure your income comfortably supports your household responsibilities before applying."
        )

    # --------------------------------------------------------
    # Positive feedback
    # --------------------------------------------------------

    if not suggestions:
        if prediction_status == "Approved":
            suggestions.append(
                "Your financial profile does not indicate any major improvement areas based on the available information."
            )
        else:
            suggestions.extend([
                "Consider reducing the requested loan amount where appropriate.",
                "Maintain or improve repayment capacity.",
                "Maintain stable and consistent income.",
                "Review recurring monthly financial obligations."
            ])

    return suggestions