from pydantic import BaseModel, Field, field_validator


class NTCApplication(BaseModel):
    dependents: int = Field(ge=0, le=3)

    employment_type: str

    annual_income: int = Field(gt=0)

    monthly_expenses: int = Field(ge=0)

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
                f"employment_type must be one of: {sorted(allowed_values)}"
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
                f"education must be one of: {sorted(allowed_values)}"
            )

        return value


class NTCPredictionResponse(BaseModel):
    prediction: str
    approved_probability: float
    rejected_probability: float
    monthly_income: float
    disposable_income: float
    expense_ratio: float
    suggestions: list[str]