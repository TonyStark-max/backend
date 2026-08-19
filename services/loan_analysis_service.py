from typing import Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_BUNDLE_PATH = BASE_DIR / "model" / "loan_model.pkl"

MODEL_ANALYSIS_THRESHOLD: float = 0.50

def evaluate_candidates(
    candidate_amounts: list[int],
    base_record: dict[str, Any],
    model: Any,
    encoders: dict[str, Any],
    features: list[str]
) -> dict[int, float]:
    """
    Evaluates a list of candidate loan amounts against the ML model.
    Returns a dictionary mapping amount -> approved_probability.
    """
    if not candidate_amounts:
        return {}
        
    df = pd.DataFrame([
        {
            **base_record,
            "Loan_Amount": amt,
        }
        for amt in candidate_amounts
    ])

    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    df = df[features]

    classes = list(model.classes_)
    approved_index = classes.index(1) if 1 in classes else 1
    
    probs = model.predict_proba(df)
    
    results = {}
    for i, amt in enumerate(candidate_amounts):
        results[amt] = float(probs[i][approved_index])
        
    return results


def generate_loan_amount_analysis(
    application_data: dict[str, Any],
    model: Any,
    encoders: dict[str, Any] | None = None,
    features: list[str] | None = None,
    threshold: float = MODEL_ANALYSIS_THRESHOLD,
) -> dict[str, Any]:
    """
    Bidirectional loan capacity estimation:
    1. If requested amount is APPROVED (prob >= threshold):
       Searches UPWARD to determine the true Maximum Predicted Eligible Loan Amount.
    2. If requested amount is REJECTED (prob < threshold):
       Searches DOWNWARD to determine how much loan amount can be approved (reduced).
    """
    requested_amount = int(application_data["loan_amount"])
    annual_income = int(application_data["annual_income"])
    credit_score = float(application_data["credit_score"])
    loan_tenure = int(application_data["loan_tenure"])

    if encoders is None or features is None:
        model_bundle = joblib.load(MODEL_BUNDLE_PATH)
        if encoders is None:
            encoders = model_bundle["encoders"]
        if features is None:
            features = model_bundle["features"]

    base_record = {
         "Dependents": application_data["dependents"],
         "Employment_Type": application_data["employment_type"],
         "Credit_Score": credit_score,
         "Annual_Income": annual_income,
         "Loan_Tenure": loan_tenure,
         "Education": application_data["education"],
    }
    
    all_evaluated: dict[int, float] = {}
    
    # 1. Baseline Evaluation at requested amount
    baseline_res = evaluate_candidates([requested_amount], base_record, model, encoders, features)
    all_evaluated.update(baseline_res)
    current_approved_prob = baseline_res[requested_amount]
    is_currently_eligible = current_approved_prob >= threshold

    # Reasonable max ceiling based on financial guidelines (e.g. 50% FOIR at 9.5% over tenure or 10x income)
    monthly_income = annual_income / 12
    max_emi_ratio = 0.50 if credit_score >= 700 else 0.40 if credit_score >= 600 else 0.30
    monthly_rate = 0.095 / 12
    n_months = max(loan_tenure * 12, 12)
    max_dti_loan = int((monthly_income * max_emi_ratio * ((1 + monthly_rate)**n_months - 1)) / (monthly_rate * (1 + monthly_rate)**n_months))
    upper_search_ceiling = max(requested_amount * 3, min(max(max_dti_loan, requested_amount * 2), 50000000))

    if is_currently_eligible:
        # ========================================================
        # CASE 1: APPROVED -> SEARCH UPWARD FOR MAXIMUM CAPACITY
        # ========================================================
        mode = "UPWARD_CAPACITY"
        
        # Build upward candidate list
        upward_set = set()
        multipliers = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
        for m in multipliers:
            amt = int(requested_amount * m)
            if amt >= 1000000:
                amt = (amt // 50000) * 50000
            elif amt >= 100000:
                amt = (amt // 10000) * 10000
            if amt > requested_amount and amt <= upper_search_ceiling:
                upward_set.add(amt)

        standard_landmarks = [
            500000, 1000000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000,
            7500000, 10000000, 15000000, 20000000, 30000000, 50000000
        ]
        for lm in standard_landmarks:
            if requested_amount < lm <= upper_search_ceiling:
                upward_set.add(lm)

        coarse_candidates = sorted(list(upward_set))
        if coarse_candidates:
            coarse_res = evaluate_candidates(coarse_candidates, base_record, model, encoders, features)
            all_evaluated.update(coarse_res)
            
            # Find the highest amount that is still approved
            highest_approved = requested_amount
            lowest_rejected = None
            
            for amt in coarse_candidates:
                if coarse_res[amt] >= threshold:
                    highest_approved = amt
                else:
                    if lowest_rejected is None:
                        lowest_rejected = amt

            # Fine search between highest_approved and lowest_rejected
            if lowest_rejected is not None and (lowest_rejected - highest_approved) > 50000:
                fine_span = lowest_rejected - highest_approved
                fine_step = max(50000, (fine_span // 6 // 50000) * 50000)
                fine_candidates = []
                cand = highest_approved + fine_step
                while cand < lowest_rejected:
                    fine_candidates.append(cand)
                    cand += fine_step

                if fine_candidates:
                    fine_res = evaluate_candidates(fine_candidates, base_record, model, encoders, features)
                    all_evaluated.update(fine_res)
                    for amt in sorted(fine_candidates):
                        if fine_res[amt] >= threshold:
                            highest_approved = amt

            recommended_amount = highest_approved
            recommended_prob = all_evaluated[recommended_amount]
        else:
            recommended_amount = requested_amount
            recommended_prob = current_approved_prob

        max_loan_status = "eligible"
        if recommended_amount > requested_amount:
            additional = recommended_amount - requested_amount
            max_loan_message = f"Based on your strong applicant profile, the ML model predicts approval for your requested amount of ₹{requested_amount:,} and estimates you could qualify for up to ₹{recommended_amount:,} (₹{additional:,} additional capacity)."
        else:
            max_loan_message = f"Based on your current applicant profile, the existing ML model predicts approval for your requested amount of ₹{requested_amount:,}."

    else:
        # ========================================================
        # CASE 2: REJECTED -> SEARCH DOWNWARD FOR REDUCED AMOUNT
        # ========================================================
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
        coarse_res = evaluate_candidates(coarse_candidates, base_record, model, encoders, features)
        all_evaluated.update(coarse_res)
        
        favorable_boundary = None
        unfavorable_boundary = requested_amount

        for amt in coarse_candidates:
            if coarse_res[amt] >= threshold:
                favorable_boundary = amt
                break
            else:
                unfavorable_boundary = amt

        # Fine search between favorable_boundary and unfavorable_boundary
        if favorable_boundary is not None and (unfavorable_boundary - favorable_boundary) > 10000:
            fine_span = unfavorable_boundary - favorable_boundary
            fine_step = max(10000, (fine_span // 6 // 10000) * 10000)
            fine_candidates = []
            cand = favorable_boundary + fine_step
            while cand < unfavorable_boundary:
                fine_candidates.append(cand)
                cand += fine_step

            if fine_candidates:
                fine_res = evaluate_candidates(fine_candidates, base_record, model, encoders, features)
                all_evaluated.update(fine_res)
                for amt in sorted(fine_candidates, reverse=True):
                    if fine_res[amt] >= threshold:
                        favorable_boundary = amt
                        break

        if favorable_boundary is not None:
            recommended_amount = favorable_boundary
            recommended_prob = all_evaluated[recommended_amount]
            max_loan_status = "eligible"
            reduction = requested_amount - favorable_boundary
            max_loan_message = f"Your requested amount of ₹{requested_amount:,} is above the model's predicted eligible limit. Based on your applicant profile, the existing ML model predicts approval up to approximately ₹{favorable_boundary:,} (Suggested reduction: ₹{reduction:,})."
        else:
            recommended_amount = None
            recommended_prob = 0.0
            max_loan_status = "none_eligible"
            max_loan_message = "Based on your current applicant profile, the existing ML model does not predict loan approval for any evaluated loan amount."

    # --- FORMAT SCENARIOS ---
    sorted_amts = sorted(all_evaluated.keys())
    
    key_amts = {requested_amount}
    if recommended_amount is not None:
        key_amts.add(recommended_amount)
    
    # Select clean subset of scenarios for display
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
        prob = all_evaluated[amt]
        scenarios.append({
            "loanAmount": amt,
            "approvalProbability": round(prob * 100, 2),
            "status": "ELIGIBLE" if prob >= threshold else "NOT_ELIGIBLE"
        })

    return {
        "mode": mode,
        "currentAmount": requested_amount,
        "recommendedAmount": recommended_amount,
        "recommendedApprovalProbability": round(recommended_prob * 100, 2),
        "threshold": round(threshold * 100, 2),
        "scenarios": scenarios,
        "max_loan_status": max_loan_status,
        "max_loan_message": max_loan_message,
    }
