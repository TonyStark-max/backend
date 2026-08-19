from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Compatibility shims for loading scikit-learn 1.6.1 serialized pipelines
# across varied Python/scikit-learn runtime versions without altering model artifacts.
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    import sklearn._loss
    if not hasattr(sklearn._loss, "CyHalfBinomialLoss"):
        sklearn._loss.CyHalfBinomialLoss = getattr(
            sklearn._loss, "HalfBinomialLoss", None
        )
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = sklearn._loss
except Exception:
    pass

try:
    import sklearn.impute._base
    if not hasattr(sklearn.impute._base.SimpleImputer, "_fill_dtype"):
        sklearn.impute._base.SimpleImputer._fill_dtype = property(
            lambda self: getattr(self, "_fit_dtype", np.float64)
        )
except Exception:
    pass

try:
    import shap
except Exception:  # pragma: no cover - fallback for broken optional dependency
    shap = None

from services.ntc_suggestion_service import generate_ntc_suggestions
from services.explainability_service import (
    FEATURE_METADATA,
    MODEL_DISCLAIMER,
    classify_impact,
    format_user_value,
    generate_feature_explanation,
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "ntc_pipeline.pkl"
MAPPING_PATH = BASE_DIR / "model" / "target_mapping.pkl"


pipeline = joblib.load(MODEL_PATH)
target_mapping = joblib.load(MAPPING_PATH)

reverse_mapping = {
    value: key
    for key, value in target_mapping.items()
}


def _compute_ntc_tree_contributions(model: Any, transformed_data: Any, feature_names: list[str]) -> list[dict]:
    X_sample = transformed_data.toarray() if hasattr(transformed_data, "toarray") else np.asarray(transformed_data)
    n_features = len(feature_names)
    contributions = np.zeros(n_features, dtype=float)
    learning_rate = getattr(model, "learning_rate", 0.1)

    for stage in getattr(model, "estimators_", []):
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

    explanation = [
        {"feature": feature_names[i], "impact": round(float(contributions[i]), 6)}
        for i in range(n_features)
    ]
    explanation.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return explanation[:10]


def _get_shap_explanation(input_data: pd.DataFrame) -> list[dict]:
    """
    Generate feature contributions for the NTC prediction.

    The input is transformed using the exact preprocessor
    stored inside ntc_pipeline.pkl.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed_data = preprocessor.transform(input_data)
        feature_names = list(preprocessor.get_feature_names_out())

        if shap is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed_data)
            if isinstance(shap_values, list):
                values = shap_values[1]
            else:
                values = shap_values
            values = values[0]

            explanation = []
            for feature_name, value in zip(feature_names, values):
                explanation.append(
                    {
                        "feature": feature_name,
                        "impact": round(float(value), 6),
                    }
                )
            explanation.sort(
                key=lambda item: abs(item["impact"]),
                reverse=True,
            )
            return explanation[:10]
        else:
            return _compute_ntc_tree_contributions(model, transformed_data, feature_names)
    except Exception:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            transformed_data = preprocessor.transform(input_data)
            feature_names = list(preprocessor.get_feature_names_out())
            return _compute_ntc_tree_contributions(model, transformed_data, feature_names)
        except Exception:
            return []


def evaluate_ntc_candidates(candidates: list[int], base_data: dict, pipeline_model) -> dict[int, float]:
    if not candidates:
        return {}
    
    # Build dataframe
    df = pd.DataFrame([
        {
            "Education": base_data["education"],
            "Dependents": base_data["dependents"],
            "Employment_Type": base_data["employment_type"],
            "Annual_Income": base_data["annual_income"],
            "Monthly_Expenses": base_data["monthly_expenses"],
            "Loan_Amount": amt,
            "Loan_Tenure": base_data["loan_tenure"],
        }
        for amt in candidates
    ])
    
    probs = pipeline_model.predict_proba(df)
    
    classes = list(pipeline_model.classes_)
    approved_index = classes.index("Approved") if "Approved" in classes else 1
    
    results = {}
    for i, amt in enumerate(candidates):
        results[amt] = float(probs[i][approved_index])
        
    return results


def predict_ntc(data: dict):
    input_data = pd.DataFrame([{
        "Education": data["education"],
        "Dependents": data["dependents"],
        "Employment_Type": data["employment_type"],
        "Annual_Income": data["annual_income"],
        "Monthly_Expenses": data["monthly_expenses"],
        "Loan_Amount": data["loan_amount"],
        "Loan_Tenure": data["loan_tenure"],
    }])

    prediction_raw = pipeline.predict(input_data)[0]
    
    if isinstance(prediction_raw, str):
        loan_status = prediction_raw
    else:
        loan_status = reverse_mapping[int(prediction_raw)]

    probabilities = pipeline.predict_proba(
        input_data
    )[0]

    classes = list(pipeline.classes_)
    approved_index = classes.index("Approved") if "Approved" in classes else 1
    rejected_index = classes.index("Rejected") if "Rejected" in classes else 0

    approved_probability = float(
        probabilities[approved_index]
    )

    rejected_probability = float(
        probabilities[rejected_index]
    )

    confidence = max(
        approved_probability,
        rejected_probability,
    )

    shap_explanation = _get_shap_explanation(
        input_data
    )

    contribution_map: dict[str, float] = {}

    for item in shap_explanation:
        raw_feature = item["feature"]
        contribution = float(item["impact"])

        if raw_feature.startswith("numeric__"):
            feature_name = raw_feature.replace("numeric__", "")
        elif raw_feature.startswith("categorical__"):
            suffix = raw_feature.replace("categorical__", "")
            if suffix.startswith("Employment_Type_"):
                feature_name = "Employment_Type"
                selected_value = data["employment_type"]
                active_value = suffix.replace("Employment_Type_", "")
                if active_value != selected_value:
                    continue
            elif suffix.startswith("Education_"):
                feature_name = "Education"
                selected_value = data["education"]
                active_value = suffix.replace("Education_", "")
                if active_value != selected_value:
                    continue
            else:
                continue
        else:
            feature_name = raw_feature

        contribution_map[feature_name] = contribution

    feature_order = [
        "Dependents",
        "Employment_Type",
        "Annual_Income",
        "Monthly_Expenses",
        "Loan_Amount",
        "Loan_Tenure",
        "Education",
    ]

    aggregated_factors = []
    for feature_name in feature_order:
        if feature_name not in contribution_map:
            continue

        contribution = contribution_map[feature_name]
        impact_level, impact_direction = classify_impact(contribution)
        formatted_value = format_user_value(feature_name, data.get(feature_name.lower() if feature_name == "Dependents" else feature_name))

        if feature_name == "Dependents":
            formatted_value = format_user_value("Dependents", data["dependents"])
        elif feature_name == "Employment_Type":
            formatted_value = data["employment_type"]
        elif feature_name == "Education":
            formatted_value = data["education"]
        elif feature_name == "Annual_Income":
            formatted_value = format_user_value("Annual_Income", data["annual_income"])
        elif feature_name == "Monthly_Expenses":
            formatted_value = format_user_value("Monthly_Expenses", data["monthly_expenses"])
        elif feature_name == "Loan_Amount":
            formatted_value = format_user_value("Loan_Amount", data["loan_amount"])
        elif feature_name == "Loan_Tenure":
            formatted_value = format_user_value("Loan_Tenure", data["loan_tenure"])

        meta = FEATURE_METADATA.get(feature_name, {})
        factor = {
            "feature": feature_name,
            "label": meta.get("label", feature_name),
            "user_value": formatted_value,
            "impact_level": impact_level,
            "impact_direction": impact_direction,
            "raw_contribution": round(float(contribution), 4),
            "explanation": generate_feature_explanation(
                feature_name,
                formatted_value,
                impact_level,
                impact_direction,
            ),
            "is_actionable": meta.get("actionable", False),
        }
        aggregated_factors.append(factor)

    negative_factors = sorted(
        [factor for factor in aggregated_factors if factor["impact_direction"] == "negative"],
        key=lambda item: item["raw_contribution"],
    )
    positive_factors = sorted(
        [factor for factor in aggregated_factors if factor["impact_direction"] == "positive"],
        key=lambda item: item["raw_contribution"],
        reverse=True,
    )

    action_plan = []
    priority = 1
    for factor in negative_factors[:3]:
        meta = FEATURE_METADATA.get(factor["feature"], {})
        action_plan.append(
            {
                "priority": priority,
                "title": meta.get("negative_title", f"Address {factor['label']}"),
                "subtitle": meta.get("negative_subtitle", f"Review {factor['label']}"),
                "factor_label": factor["label"],
                "reason": factor["explanation"],
                "recommendation": meta.get("action_template", ""),
            }
        )
        priority += 1

    explanation = {
        "top_negative_factors": negative_factors[:3],
        "positive_factors": positive_factors[:3],
        "all_factors": aggregated_factors,
        "action_plan": action_plan,
        "disclaimer": MODEL_DISCLAIMER,
    }

    suggestions = generate_ntc_suggestions(
        data, loan_status
    )

    monthly_income = data["annual_income"] / 12
    monthly_expenses = data["monthly_expenses"]
    disposable_income = monthly_income - monthly_expenses
    expense_ratio = (monthly_expenses / monthly_income) * 100 if monthly_income > 0 else 0

    # --------------------------------------------------------
    # NTC Maximum Predicted Eligible Loan Analysis (Bidirectional)
    # --------------------------------------------------------
    requested_amount = int(data["loan_amount"])
    loan_tenure = int(data["loan_tenure"])
    threshold = 0.50
    
    all_evaluated_ntc: dict[int, float] = {requested_amount: approved_probability}

    # Max debt capacity based on disposable income & tenure
    disposable = max(0, disposable_income)
    monthly_rate = 0.095 / 12
    n_months = max(loan_tenure * 12, 12)
    max_dti_loan = int((disposable * 0.60 * ((1 + monthly_rate)**n_months - 1)) / (monthly_rate * (1 + monthly_rate)**n_months))
    upper_ceiling = max(requested_amount * 3, min(max(max_dti_loan, requested_amount * 2), 50000000))

    if loan_status == "Approved" and approved_probability >= threshold:
        # Case 1: Approved -> Search UPWARD
        mode = "UPWARD_CAPACITY"
        upward_set = set()
        multipliers = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
        for m in multipliers:
            amt = int(requested_amount * m)
            if amt >= 1000000:
                amt = (amt // 50000) * 50000
            elif amt >= 100000:
                amt = (amt // 10000) * 10000
            if amt > requested_amount and amt <= upper_ceiling:
                upward_set.add(amt)

        standard_landmarks = [
            500000, 1000000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000,
            7500000, 10000000, 15000000, 20000000, 30000000, 50000000
        ]
        for lm in standard_landmarks:
            if requested_amount < lm <= upper_ceiling:
                upward_set.add(lm)

        coarse_candidates = sorted(list(upward_set))
        if coarse_candidates:
            coarse_res = evaluate_ntc_candidates(coarse_candidates, data, pipeline)
            all_evaluated_ntc.update(coarse_res)
            
            highest_approved = requested_amount
            lowest_rejected = None
            for amt in coarse_candidates:
                if coarse_res[amt] >= threshold:
                    highest_approved = amt
                else:
                    if lowest_rejected is None:
                        lowest_rejected = amt

            if lowest_rejected is not None and (lowest_rejected - highest_approved) > 50000:
                fine_span = lowest_rejected - highest_approved
                fine_step = max(50000, (fine_span // 6 // 50000) * 50000)
                fine_candidates = []
                cand = highest_approved + fine_step
                while cand < lowest_rejected:
                    fine_candidates.append(cand)
                    cand += fine_step

                if fine_candidates:
                    fine_res = evaluate_ntc_candidates(fine_candidates, data, pipeline)
                    all_evaluated_ntc.update(fine_res)
                    for amt in sorted(fine_candidates):
                        if fine_res[amt] >= threshold:
                            highest_approved = amt

            maximum_eligible_amount = highest_approved
            max_eligible_approved_probability = all_evaluated_ntc[maximum_eligible_amount]
        else:
            maximum_eligible_amount = requested_amount
            max_eligible_approved_probability = approved_probability

        max_loan_status = "eligible"
        max_prediction = "Approved"
        if maximum_eligible_amount > requested_amount:
            additional = maximum_eligible_amount - requested_amount
            max_loan_message = f"Based on your applicant profile and disposable income, the ML model predicts approval for your requested amount of ₹{requested_amount:,} and estimates you could qualify for up to ₹{maximum_eligible_amount:,} (₹{additional:,} additional capacity)."
        else:
            max_loan_message = f"Based on your applicant profile, the ML model predicts approval for your requested amount of ₹{requested_amount:,}."

    else:
        # Case 2: Rejected -> Search DOWNWARD
        mode = "DOWNWARD_IMPROVEMENT"
        downward_set = set()
        percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
        for p in percentages:
            amt = int(requested_amount * p)
            if amt >= 1000000:
                amt = (amt // 50000) * 50000
            elif amt >= 100000:
                amt = (amt // 10000) * 10000
            else:
                amt = (amt // 5000) * 5000
            if 0 < amt < requested_amount:
                downward_set.add(amt)

        standard_landmarks = [
            50000000, 30000000, 20000000, 15000000, 10000000, 7500000, 5000000,
            4000000, 3000000, 2500000, 2000000, 1500000, 1200000, 1000000,
            800000, 750000, 600000, 500000, 400000, 300000, 200000, 100000, 50000, 25000, 10000
        ]
        for lm in standard_landmarks:
            if 0 < lm < requested_amount:
                downward_set.add(lm)

        coarse_candidates = sorted(list(downward_set), reverse=True)
        coarse_res = evaluate_ntc_candidates(coarse_candidates, data, pipeline)
        all_evaluated_ntc.update(coarse_res)
        
        favorable_boundary = None
        unfavorable_boundary = requested_amount
        for amt in coarse_candidates:
            if coarse_res[amt] >= threshold:
                favorable_boundary = amt
                break
            else:
                unfavorable_boundary = amt

        if favorable_boundary is not None and (unfavorable_boundary - favorable_boundary) > 10000:
            fine_span = unfavorable_boundary - favorable_boundary
            fine_step = max(10000, (fine_span // 6 // 10000) * 10000)
            fine_candidates = []
            cand = favorable_boundary + fine_step
            while cand < unfavorable_boundary:
                fine_candidates.append(cand)
                cand += fine_step

            if fine_candidates:
                fine_res = evaluate_ntc_candidates(fine_candidates, data, pipeline)
                all_evaluated_ntc.update(fine_res)
                for amt in sorted(fine_candidates, reverse=True):
                    if fine_res[amt] >= threshold:
                        favorable_boundary = amt
                        break

        if favorable_boundary is not None:
            maximum_eligible_amount = favorable_boundary
            max_eligible_approved_probability = all_evaluated_ntc[maximum_eligible_amount]
            max_loan_status = "eligible"
            reduction = requested_amount - favorable_boundary
            max_loan_message = f"Your requested amount of ₹{requested_amount:,} is above the model's predicted eligible limit. Based on your applicant profile, the ML model predicts approval up to approximately ₹{favorable_boundary:,} (Suggested reduction: ₹{reduction:,})."
            max_prediction = "Approved"
        else:
            maximum_eligible_amount = None
            max_eligible_approved_probability = 0.0
            max_loan_status = "none_eligible"
            max_loan_message = "Based on your current applicant profile, the existing ML model does not predict loan approval for any evaluated loan amount."
            max_prediction = "Rejected"

    # Format scenarios
    sorted_amts = sorted(all_evaluated_ntc.keys())
    key_amts = {requested_amount}
    if maximum_eligible_amount is not None:
        key_amts.add(maximum_eligible_amount)
    
    if len(sorted_amts) > 8:
        step = len(sorted_amts) / 7
        selected_amts = key_amts.copy()
        for i in range(7):
            idx = int(i * step)
            if idx < len(sorted_amts):
                selected_amts.add(sorted_amts[idx])
        final_amts = sorted(list(selected_amts))
    else:
        final_amts = sorted_amts

    scenarios = []
    for amt in final_amts:
        prob = all_evaluated_ntc[amt]
        scenarios.append({
            "loanAmount": amt,
            "approvalProbability": round(prob * 100, 2),
            "status": "ELIGIBLE" if prob >= threshold else "NOT_ELIGIBLE"
        })

    loan_amount_analysis = {
        "mode": mode,
        "currentAmount": requested_amount,
        "recommendedAmount": maximum_eligible_amount,
        "recommendedApprovalProbability": round(max_eligible_approved_probability * 100, 2),
        "threshold": round(threshold * 100, 2),
        "scenarios": scenarios,
        "max_loan_status": max_loan_status,
        "max_loan_message": max_loan_message,
    }

    return {
        "prediction": loan_status,
        "confidence": round(confidence, 4),
        "approved_probability": round(
            approved_probability,
            4,
        ),
        "rejected_probability": round(
            rejected_probability,
            4,
        ),
        "requested_loan_amount": requested_amount,
        "maximum_eligible_amount": maximum_eligible_amount,
        "maximum_eligible_prediction": max_prediction,
        "max_eligible_approved_probability": round(max_eligible_approved_probability, 4),
        "max_loan_status": max_loan_status,
        "max_loan_message": max_loan_message,
        "loan_amount_analysis": loan_amount_analysis,
        "explanation": explanation,
        "shap_explanation": shap_explanation,
        "suggestions": suggestions,
        "monthly_income": monthly_income,
        "disposable_income": disposable_income,
        "expense_ratio": expense_ratio,
    }
