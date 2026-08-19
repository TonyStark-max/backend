from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    dependents: int = Field(ge=0, le=3)

    employment_type: str

    annual_income: int = Field(gt=0)

    credit_score: float = Field(ge=300, le=900)

    loan_amount: int = Field(gt=0)

    loan_tenure: int = Field(ge=2, le=30)

    education: str

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, value: str) -> str:
        allowed_values = {
            "Private",
            "Government",
            "Self-Employed",
            "Unemployed",
            "Skilled Labor",
        }

        if value not in allowed_values:
            raise ValueError(
                "employment_type must be one of: "
                f"{sorted(allowed_values)}"
            )

        return value

    @field_validator("education")
    @classmethod
    def validate_education(cls, value: str) -> str:
        allowed_values = {
            "Graduate",
            "Post Graduate",
            "PhD",
            "High School",
            "Diploma",
            "No Formal",
        }

        if value not in allowed_values:
            raise ValueError(
                "education must be one of: "
                f"{sorted(allowed_values)}"
            )

        return value


class ExplanationFactor(BaseModel):
    feature: str
    label: str
    user_value: str
    impact_level: str
    impact_direction: str
    raw_contribution: float
    explanation: str
    is_actionable: bool


class ActionPlanItem(BaseModel):
    priority: int
    title: str
    factor_label: str
    reason: str
    recommendation: str


class LoanExplanation(BaseModel):
    top_negative_factors: list[ExplanationFactor]
    positive_factors: list[ExplanationFactor]
    all_factors: list[ExplanationFactor]
    action_plan: list[ActionPlanItem]
    disclaimer: str


class PredictionResponse(BaseModel):
    prediction: str
    approved_probability: float
    rejected_probability: float
    suggestions: list[str]
    explanation: LoanExplanation | None = None
    loan_amount_analysis: dict | None = None
    requested_loan_amount: int | None = None
    maximum_eligible_amount: int | None = None
    maximum_eligible_prediction: str | None = None
    max_eligible_approved_probability: float | None = None
    max_loan_status: str | None = None
    max_loan_message: str | None = None


class MaxLoanEstimateResponse(BaseModel):
    requested_loan_amount: int
    maximum_eligible_amount: int | None = None
    maximum_eligible_prediction: str
    max_eligible_approved_probability: float
    max_loan_status: str
    max_loan_message: str
    loan_amount_analysis: dict | None = None


class ValidationResponse(BaseModel):
    status: str
    message: str
    data: LoanApplication