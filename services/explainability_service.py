import numpy as np
import pandas as pd
from typing import Any


FEATURE_METADATA = {
    "Credit_Score": {
        "label": "Credit Score (CIBIL)",
        "unit": "score points",
        "actionable": True,
        "negative_title": "Strengthen your credit profile",
        "negative_subtitle": "Improve CIBIL score & payment history",
        "action_template": (
            "Review your credit report for any discrepancies or outdated records. "
            "Prioritize consistent, on-time repayments for all current debts and keep credit "
            "utilization within moderate limits before submitting a future application."
        ),
    },
    "Loan_Amount": {
        "label": "Requested Loan Amount",
        "unit": "INR",
        "actionable": True,
        "negative_title": "Review requested loan amount",
        "negative_subtitle": "Review the amount you're requesting",
        "action_template": (
            "Evaluate whether the requested loan amount can be reduced to better match your "
            "current financial profile, verifiable income, and overall repayment capacity."
        ),
    },
    "Annual_Income": {
        "label": "Annual Income",
        "unit": "INR",
        "actionable": True,
        "negative_title": "Strengthen demonstrated income & affordability",
        "negative_subtitle": "Strengthen documented affordability",
        "action_template": (
            "Ensure all regular and verifiable income sources are fully documented. "
            "Consider reviewing loan affordability relative to your net disposable income or "
            "applying alongside an eligible co-applicant where permitted."
        ),
    },
    "Loan_Tenure": {
        "label": "Loan Tenure",
        "unit": "years",
        "actionable": True,
        "negative_title": "Evaluate repayment tenure options",
        "negative_subtitle": "Review your repayment period",
        "action_template": (
            "Review the available loan tenure options and consider selecting a repayment term "
            "that balances affordable monthly installments with your overall financial commitments."
        ),
    },
    "Employment_Type": {
        "label": "Employment Profile",
        "unit": "category",
        "actionable": True,
        "negative_title": "Provide verified employment & income continuity",
        "negative_subtitle": "Build verified employment continuity",
        "action_template": (
            "Ensure a consistent track record of stable employment or continuous verifiable business "
            "cash flows with organized income proofs before applying again."
        ),
    },
    "Dependents": {
        "label": "Household Dependents",
        "unit": "dependents",
        "actionable": False,
        "negative_title": "Account for household commitments in budget",
        "negative_subtitle": "Optimize disposable savings",
        "action_template": (
            "While household size is a personal background factor, maintaining higher disposable "
            "savings buffers after family living expenses can improve financial resilience."
        ),
    },
    "Education": {
        "label": "Education Level",
        "unit": "category",
        "actionable": False,
        "negative_title": "Focus on controllable financial metrics",
        "negative_subtitle": "Strengthen core financial health",
        "action_template": (
            "Educational background is a fixed profile attribute. Focus future preparation on "
            "controllable financial health metrics such as credit score and loan sizing."
        ),
    },
}

MODEL_DISCLAIMER = (
    "This explanation describes how the current machine-learning model responded to the "
    "information provided for this assessment. It does not represent a guaranteed or legally "
    "binding explanation of a lender's decision, and the recommended actions do not guarantee "
    "future loan approval."
)


def format_user_value(feature_name: str, value: Any) -> str:
    """Format raw feature value into clear human-readable string."""
    if feature_name in ("Loan_Amount", "Annual_Income"):
        try:
            num = int(value)
            return f"₹{num:,}"
        except Exception:
            return f"₹{value}"
    elif feature_name == "Credit_Score":
        try:
            return f"{float(value):.0f}"
        except Exception:
            return str(value)
    elif feature_name == "Loan_Tenure":
        return f"{value} years"
    elif feature_name == "Dependents":
        count = int(value) if str(value).isdigit() else value
        return f"{count} {'dependent' if count == 1 else 'dependents'}"
    else:
        return str(value)


def classify_impact(contribution: float) -> tuple[str, str]:
    """
    Classify the magnitude and direction of feature contribution.
    Returns: (impact_level, impact_direction)
    """
    if contribution < -0.8:
        return "Strong negative influence", "negative"
    elif contribution < -0.2:
        return "Moderate negative influence", "negative"
    elif contribution < 0:
        return "Low negative influence", "negative"
    elif contribution == 0:
        return "Neutral influence", "neutral"
    elif contribution <= 0.2:
        return "Low positive influence", "positive"
    elif contribution <= 0.8:
        return "Moderate positive influence", "positive"
    else:
        return "Strong positive influence", "positive"


# Rich, varied humanized explanations per feature and impact level
EXPLANATION_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "Loan_Amount": {
        "negative": {
            "Strong": "The amount you requested was significantly higher than what the model associates with supported eligibility for this profile.",
            "Moderate": "The amount you requested was one of the factors working against this assessment.",
            "Low": "The loan amount you entered had a noticeable negative influence on this result.",
        },
        "positive": {
            "Strong": "The requested loan amount was modest and provided strong positive support for eligibility.",
            "Moderate": "The requested loan amount was within a supportive range for this assessment.",
            "Low": "The requested loan amount had a minor positive influence.",
        },
    },
    "Loan_Tenure": {
        "negative": {
            "Strong": "The selected repayment period created a substantial imbalance in the model's repayment evaluation.",
            "Moderate": "The selected repayment period had a negative influence on the model's assessment.",
            "Low": "The requested repayment timeframe was relatively less supportive in this assessment.",
        },
        "positive": {
            "Strong": "Your chosen repayment tenure was well-balanced and strongly supported this result.",
            "Moderate": "Your selected repayment period was supportive in this assessment.",
            "Low": "The loan tenure provided a minor positive contribution.",
        },
    },
    "Annual_Income": {
        "negative": {
            "Strong": "Your reported income was significantly lower than what the model seeks to support the requested credit profile.",
            "Moderate": "Your reported income was less supportive of eligibility in this particular assessment.",
            "Low": "Your income level had a slight negative influence relative to the overall application.",
        },
        "positive": {
            "Strong": "Your reported annual income provided strong positive support for this assessment.",
            "Moderate": "Your reported annual income positively supported this assessment.",
            "Low": "Your annual income contributed slightly in your favor.",
        },
    },
    "Credit_Score": {
        "negative": {
            "Strong": "Your credit profile was one of the primary factors pulling down this assessment.",
            "Moderate": "Your credit score was lower than what the model typically looks for to support eligibility.",
            "Low": "Your credit profile was slightly below the optimal range for this assessment.",
        },
        "positive": {
            "Strong": "Your credit profile was one of the strongest factors supporting this assessment.",
            "Moderate": "Your credit score was one of the factors working in your favor.",
            "Low": "Your credit profile helped support this assessment, although its influence was relatively small.",
        },
    },
    "Employment_Type": {
        "negative": {
            "Strong": "Your current employment profile was one of the major factors working against this assessment.",
            "Moderate": "Your employment status had a noticeable negative influence on the eligibility score.",
            "Low": "Your employment category had a slight negative influence on this assessment.",
        },
        "positive": {
            "Strong": "Your employment stability was a major positive contributor to this assessment.",
            "Moderate": "Your employment background was a supportive factor in this assessment.",
            "Low": "Your employment profile had a slight positive influence.",
        },
    },
    "Dependents": {
        "negative": {
            "Strong": "Household commitments relative to income had a strong negative influence on this evaluation.",
            "Moderate": "The number of household dependents weighed moderately against this result.",
            "Low": "Household size had a slight negative influence on the overall score.",
        },
        "positive": {
            "Strong": "Having minimal household dependents provided strong financial resilience in the assessment.",
            "Moderate": "Household size was supportive of repayment capacity in this assessment.",
            "Low": "Household dependent count had a minor positive influence.",
        },
    },
    "Education": {
        "negative": {
            "Strong": "Educational background contributed negatively to the model's statistical weighting.",
            "Moderate": "Educational profile had a moderate negative influence on this assessment.",
            "Low": "Educational background had a minor negative weighting in this result.",
        },
        "positive": {
            "Strong": "Your educational qualifications provided strong positive weighting in the assessment.",
            "Moderate": "Your education level was a positive contributing factor.",
            "Low": "Educational background had a minor positive contribution.",
        },
    },
}


def generate_feature_explanation(
    feature_name: str,
    formatted_value: str,
    impact_level: str,
    impact_direction: str,
) -> str:
    """Generate model influence explanation in natural human language."""
    level_key = "Moderate"
    if "Strong" in impact_level:
        level_key = "Strong"
    elif "Low" in impact_level:
        level_key = "Low"

    feat_templates = EXPLANATION_TEMPLATES.get(feature_name, {})
    dir_templates = feat_templates.get(impact_direction, {})

    if level_key in dir_templates:
        return dir_templates[level_key]

    meta = FEATURE_METADATA.get(feature_name, {})
    label = meta.get("label", feature_name)

    if impact_direction == "negative":
        return f"{label} was less supportive of eligibility in this assessment."
    elif impact_direction == "positive":
        return f"{label} contributed favorably toward this assessment."
    else:
        return f"{label} had a neutral influence on this assessment."


def compute_tree_path_contributions(
    model: Any,
    encoded_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """
    Compute exact local feature contributions for a GradientBoostingClassifier
    using tree path attribution (Saabas algorithm).
    """
    X_sample = encoded_df[features].values
    n_features = len(features)
    contributions = np.zeros(n_features, dtype=float)
    learning_rate = getattr(model, "learning_rate", 0.1)

    for stage in model.estimators_:
        tree = stage[0].tree_
        node = 0
        while tree.feature[node] >= 0:
            feat_idx = tree.feature[node]
            thresh = tree.threshold[node]
            current_val = tree.value[node, 0, 0]

            if X_sample[0, feat_idx] <= thresh:
                next_node = tree.children_left[node]
            else:
                next_node = tree.children_right[node]

            next_val = tree.value[next_node, 0, 0]
            contributions[feat_idx] += learning_rate * (next_val - current_val)
            node = next_node

    return {features[i]: float(contributions[i]) for i in range(n_features)}


def explain_prediction(
    application_data: dict[str, Any],
    encoded_df: pd.DataFrame,
    model: Any,
    features: list[str],
    is_approved: bool,
) -> dict[str, Any]:
    """
    Generate dynamic local feature contributions and a personalized approval
    action plan for the specific applicant's prediction.
    """
    # 1. Compute exact local contributions
    contributions = compute_tree_path_contributions(model, encoded_df, features)

    # Feature key mapping from snake_case application_data to model feature names
    key_mapping = {
        "Dependents": application_data.get("dependents"),
        "Employment_Type": application_data.get("employment_type"),
        "Credit_Score": application_data.get("credit_score"),
        "Annual_Income": application_data.get("annual_income"),
        "Loan_Amount": application_data.get("loan_amount"),
        "Loan_Tenure": application_data.get("loan_tenure"),
        "Education": application_data.get("education"),
    }

    all_factors = []
    negative_factors = []
    positive_factors = []

    for feat in features:
        raw_val = key_mapping.get(feat)
        formatted_val = format_user_value(feat, raw_val)
        contrib_val = contributions.get(feat, 0.0)
        impact_level, impact_dir = classify_impact(contrib_val)
        meta = FEATURE_METADATA.get(feat, {})
        explanation_text = generate_feature_explanation(
            feat, formatted_val, impact_level, impact_dir
        )

        factor_obj = {
            "feature": feat,
            "label": meta.get("label", feat),
            "user_value": formatted_val,
            "impact_level": impact_level,
            "impact_direction": impact_dir,
            "raw_contribution": round(contrib_val, 4),
            "explanation": explanation_text,
            "is_actionable": meta.get("actionable", False),
        }

        all_factors.append(factor_obj)

        if impact_dir == "negative":
            negative_factors.append(factor_obj)
        elif impact_dir == "positive":
            positive_factors.append(factor_obj)

    # Rank negative factors from most negative to least negative
    negative_factors.sort(key=lambda x: x["raw_contribution"])

    # Rank positive factors from most positive to least positive
    positive_factors.sort(key=lambda x: x["raw_contribution"], reverse=True)

    # 2. Generate personalized action plan
    # Filter actionable negative factors first, falling back to general negative factors
    actionable_negatives = [f for f in negative_factors if f["is_actionable"]]
    
    action_plan: list[dict[str, Any]] = []
    priority_count = 1

    for factor in actionable_negatives[:3]:
        feat = factor["feature"]
        meta = FEATURE_METADATA.get(feat, {})
        action_plan.append(
            {
                "priority": priority_count,
                "title": meta.get("negative_title", f"Address {factor['label']}"),
                "subtitle": meta.get("negative_subtitle", f"Review {factor['label']}"),
                "factor_label": factor["label"],
                "reason": factor["explanation"],
                "recommendation": meta.get("action_template", ""),
            }
        )
        priority_count += 1

    # In case there were non-actionable negative factors or few negatives
    if not action_plan and not is_approved:
        # If there are general negative factors
        for factor in negative_factors[:3]:
            feat = factor["feature"]
            meta = FEATURE_METADATA.get(feat, {})
            action_plan.append(
                {
                    "priority": priority_count,
                    "title": meta.get("negative_title", f"Address {factor['label']}"),
                    "subtitle": meta.get("negative_subtitle", f"Review {factor['label']}"),
                    "factor_label": factor["label"],
                    "reason": factor["explanation"],
                    "recommendation": meta.get("action_template", ""),
                }
            )
            priority_count += 1

    return {
        "top_negative_factors": negative_factors[:3],
        "positive_factors": positive_factors[:3],
        "all_factors": all_factors,
        "action_plan": action_plan,
        "disclaimer": MODEL_DISCLAIMER,
    }
