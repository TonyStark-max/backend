import os
import json
from typing import Any, List, Optional
import httpx
import re

SYSTEM_INSTRUCTION = """You are "LoanWise AI", an expert Loan & Financial Assistant specializing in:
1. Loan Approval Prediction & Eligibility factors (Annual Income, CIBIL/Credit Score, DTI ratio, Employment stability, Dependents, Loan Tenure).
2. Credit Score (CIBIL/Experian) Guidance: Score ranges (300-900), improving credit score from poor (<600) to fair (600-699) and good/excellent (700-900), dispute resolution, credit utilization ratio (<30%), impact of multiple hard inquiries, and credit mix.
3. Loan Types & Terms: Personal loans, Home loans, Education loans, Vehicle loans, Business loans, Gold loans, Fixed vs Floating interest rates.
4. EMI Planning & Debt Management: EMI calculation formula, balancing tenure vs total interest paid, prepayment strategies, foreclosure, and Debt-to-Income (DTI) optimization.
5. Documentation & Verification: KYC (Aadhaar, PAN), income proof (ITR, Form 16, Salary Slips, Bank Statements), property documents, and guarantor requirements.
6. Core Financial Literacy: Money, interest rates, inflation, compound interest, credit cards vs debit cards, collateral, amortization, and tax benefits.

Tone and Style:
- Direct, clear, professional, educational, and structured with clear headings, bullet points, bold key terms, and actionable takeaways.
- Provide immediate, direct answers to the exact question asked without generic deflection.
- If the user asks a question that is NOT related to finance or loans, you MUST STILL answer it helpfully and accurately as a general AI assistant. Do not refuse to answer off-topic questions.
- When relevant, offer tips on how to improve borrowing outcomes.
"""

KNOWLEDGE_BASE = [
    {
        "keywords": ["what is money", "meaning of money", "define money", "concept of money"],
        "reply": (
            "### 💵 What is Money?\n\n"
            "**Money** is any universally accepted medium of exchange that allows individuals and businesses to trade goods, services, and settle debts without relying on direct barter.\n\n"
            "#### The 4 Core Functions of Money:\n"
            "1. **Medium of Exchange**: Facilitates smooth trade without requiring a 'double coincidence of wants'.\n"
            "2. **Unit of Account**: Provides a common metric (like ₹ INR or $ USD) to measure the economic value of goods and assets.\n"
            "3. **Store of Value**: Enables wealth to be saved and retrieved in the future (though inflation can erode purchasing power over time).\n"
            "4. **Standard of Deferred Payment**: Forms the legal foundation for borrowing, loans, credit, and future contracts.\n\n"
            "#### Forms of Modern Money:\n"
            "- **Fiat Currency**: Cash issued by central banks (e.g., RBI, Federal Reserve) backed by government guarantee.\n"
            "- **Commercial Bank Money**: Digital balances in savings and current bank accounts.\n"
            "- **Digital Currency & UPI**: Electronic payment layers enabling instant settlement."
        ),
        "suggestions": [
            "What is a loan and how does it work?",
            "What is inflation and how does it affect loans?",
            "What is the difference between simple and compound interest?"
        ]
    },
    {
        "keywords": ["what is loan", "what is a loan", "meaning of loan", "define loan", "how do loans work", "what are loans"],
        "reply": (
            "### 🏦 What is a Loan and How Does It Work?\n\n"
            "A **loan** is a financial contract where a **lender** (a bank, NBFC, or credit union) gives a lump sum of money (the **Principal**) to a **borrower**, who agrees to repay that amount over a specified period (**Tenure**) along with an additional fee called **Interest**.\n\n"
            "#### Key Elements of Every Loan:\n"
            "1. **Principal Amount**: The original sum of money borrowed.\n"
            "2. **Interest Rate**: The percentage charged by the lender for the use of their capital (can be Fixed or Floating).\n"
            "3. **Loan Tenure**: The duration (months or years) allowed to pay back the loan.\n"
            "4. **EMI (Equated Monthly Installment)**: Fixed monthly payments combining part of the principal and part of the interest.\n"
            "5. **Collateral/Security**: An asset pledged by the borrower for secured loans (e.g., house, car, gold).\n\n"
            "#### Broad Categories of Loans:\n"
            "- **Secured Loans**: Backed by assets (Home Loans, Auto Loans, Gold Loans). Feature lower interest rates and higher sanction amounts.\n"
            "- **Unsecured Loans**: No collateral required (Personal Loans, Education Loans, Credit Cards). Rely heavily on your **CIBIL Score** and income stability."
        ),
        "suggestions": [
            "How do banks calculate loan eligibility?",
            "What is the difference between secured and unsecured loans?",
            "What is an EMI and how is it calculated?"
        ]
    },
    {
        "keywords": ["what is interest", "interest rate", "how interest works", "simple vs compound interest", "apr"],
        "reply": (
            "### 📈 Understanding Interest Rates and How They Work\n\n"
            "**Interest** is the cost of borrowing money. From the borrower's perspective, it is the rent paid to use the lender's funds; from the lender's perspective, it is the reward for taking on credit risk.\n\n"
            "#### 1. Simple vs. Compound Interest:\n"
            "- **Simple Interest (SI)**: Calculated purely on the initial principal amount (`SI = P × R × T / 100`).\n"
            "- **Compound Interest (CI)**: Interest is calculated on the initial principal *plus* all accumulated interest from previous periods ('interest on interest'). Almost all retail loans and credit cards use reducing-balance compounding.\n\n"
            "#### 2. Fixed vs. Floating Interest Rate:\n"
            "- **Fixed Rate**: The interest rate remains locked throughout the entire loan tenure regardless of market fluctuations.\n"
            "- **Floating / Variable Rate**: The rate is linked to an external benchmark (like the RBI Repo Rate) and adjusts periodically when central bank policy changes.\n\n"
            "#### 3. What is APR (Annual Percentage Rate)?\n"
            "APR represents the true total annual cost of the loan, including the base interest rate plus upfront processing fees, administrative charges, and documentation costs."
        ),
        "suggestions": [
            "Should I choose fixed or floating interest rate?",
            "How does credit score impact my interest rate?",
            "How to reduce total interest paid on a loan?"
        ]
    },
    {
        "keywords": ["fixed vs floating", "fixed rate", "floating rate", "variable rate"],
        "reply": (
            "### ⚖️ Fixed Rate vs. Floating Rate Loans: Which Should You Choose?\n\n"
            "| Feature | Fixed Interest Rate | Floating Interest Rate |\n"
            "| :--- | :--- | :--- |\n"
            "| **Monthly EMI** | Remains 100% constant throughout tenure | Fluctuates based on benchmark (RBI Repo Rate) |\n"
            "| **Budget Predictability** | High – exact monthly outflow is known | Variable – EMI or tenure may increase or decrease |\n"
            "| **Initial Interest Rate** | Usually 1.0% - 2.5% higher than floating | Typically lower initial rate |\n"
            "| **Prepayment Penalties** | Banks may charge 2-4% penalty on fixed loans | **Zero penalty** for individual floating home loans (RBI mandate) |\n\n"
            "**💡 Recommendation:**\n"
            "- Choose **Floating Rate** for long-term loans (e.g. Home Loans, 10-20 yrs) to benefit from rate cut cycles and penalty-free prepayments.\n"
            "- Choose **Fixed Rate** during historically low-interest regimes or for short-term personal/auto loans where budget certainty is paramount."
        ),
        "suggestions": [
            "How does RBI repo rate affect home loan EMIs?",
            "What is prepayment penalty on personal loans?",
            "How to boost CIBIL score to get the lowest rate?"
        ]
    },
    {
        "keywords": ["personal loan", "unsecured loan", "what is personal loan"],
        "reply": (
            "### 💳 Personal Loans: Features, Eligibility & Pros/Cons\n\n"
            "A **Personal Loan** is an **unsecured** multi-purpose loan that you can use for medical emergencies, home renovation, weddings, travel, or debt consolidation without pledging any asset.\n\n"
            "#### Key Characteristics:\n"
            "- **Tenure**: Typically 1 to 5 years (up to 7 years with some lenders).\n"
            "- **Interest Rates**: Typically ranges between **10.5% and 24% p.a.** depending on credit score and employer profile.\n"
            "- **Sanction Speed**: Fast approval (often same-day to 48 hours) for pre-approved salaried applicants.\n\n"
            "#### Crucial Approval Factors:\n"
            "1. **CIBIL Score (750+)**: Low scores trigger high interest rates or outright rejection.\n"
            "2. **Net Monthly Salary**: Consistent monthly bank credits (minimum ₹25,000 - ₹35,000 for top banks).\n"
            "3. **Employer Reputation**: Employees of Category 'A' MNCs or Government departments receive favorable rates."
        ),
        "suggestions": [
            "What are the top reasons personal loans get rejected?",
            "Personal Loan vs Gold Loan: Which is cheaper?",
            "How to calculate personal loan EMI?"
        ]
    },
    {
        "keywords": ["home loan", "housing loan", "mortgage", "what is home loan", "property loan"],
        "reply": (
            "### 🏡 Home Loans (Mortgages): Complete Overview\n\n"
            "A **Home Loan** is a secured loan provided by banks/housing finance companies (HFCs) to purchase a ready home, construct a house, buy a plot, or renovate existing property.\n\n"
            "#### Key Features:\n"
            "- **Tenure**: Long repayment terms from **10 to 30 years**.\n"
            "- **LTV (Loan-to-Value) Ratio**: Banks fund **75% to 90%** of the property's registered value; the remaining 10-25% is paid as your **Down Payment**.\n"
            "- **Tax Deductions** (Indian Income Tax Act):\n"
            "  - **Section 80C**: Up to ₹1.5 Lakh per financial year on Principal repayment.\n  - **Section 24(b)**: Up to ₹2 Lakh per financial year on Interest paid for self-occupied property.\n\n"
            "#### Key Documents Needed:\n"
            "- Property chain title documents, sanctioned building plan, NOC from society/builder, 3-yr ITR, and 6-month salary bank statements."
        ),
        "suggestions": [
            "How to calculate maximum home loan eligibility?",
            "What is LTV (Loan-to-Value) ratio?",
            "How do partial prepayments shorten home loan tenure?"
        ]
    },
    {
        "keywords": ["credit card", "debit card", "credit card vs debit card", "difference between credit and debit"],
        "reply": (
            "### 💳 Credit Card vs. Debit Card: Key Differences\n\n"
            "| Aspect | Debit Card | Credit Card |\n"
            "| :--- | :--- | :--- |\n"
            "| **Source of Funds** | Deducted instantly from your own bank account | Borrowed from the bank's pre-approved credit line |\n"
            "| **Credit Building** | Does **NOT** build or impact your CIBIL score | Directly reports repayment history to CIBIL/Experian |\n"
            "| **Interest Grace Period** | None (your own money is used) | **45 to 50 interest-free days** if bill is paid in full |\n"
            "| **Fraud Protection** | Slower resolution if account is drained | High protection; transactions can be disputed/charged back |\n"
            "| **Perks & Rewards** | Basic reward points / cashback | Higher reward points, airport lounge access, milestones |\n\n"
            "**⚠️ Golden Rule for Credit Cards:** Always pay the **Total Amount Due** in full before the billing due date. Never pay just the 'Minimum Due' because the remaining balance accrues 36% - 45% annual interest!"
        ),
        "suggestions": [
            "How does credit card utilization affect CIBIL score?",
            "Does closing an old credit card decrease my credit score?",
            "How to boost CIBIL score to 750+?"
        ]
    },
    {
        "keywords": ["cibil", "credit score", "improve score", "increase score", "raise score", "low score", "what is cibil"],
        "reply": (
            "### 📈 How to Improve & Strengthen Your CIBIL Credit Score\n\n"
            "Your CIBIL/Credit Score (ranging between **300 and 900**) is one of the most critical factors in loan approval. "
            "A score of **750+** is generally considered ideal by most banks and NBFCs.\n\n"
            "#### Key Strategies to Boost Your Score:\n"
            "1. **Timely Repayments (35% impact)**: Always pay EMIs and credit card bills before the due date. Even a single 30-day default can lower your score by 50+ points.\n"
            "2. **Maintain Low Credit Utilization (< 30%)**: Keep your total credit card spends under 30% of your available limit.\n"
            "3. **Avoid Multiple Loan Inquiries**: Submitting multiple loan applications within a short window triggers 'Hard Inquiries', signaling credit hunger.\n"
            "4. **Maintain a Healthy Credit Mix**: Balance secured loans (home, auto) and unsecured loans (personal, credit cards).\n"
            "5. **Check for Errors on CIBIL Report**: Periodically download your official report and dispute any misreported defaults or active accounts you already closed."
        ),
        "suggestions": [
            "What is the ideal Debt-to-Income (DTI) ratio for loans?",
            "What documents are required for quick loan approval?",
            "Does checking my own CIBIL score reduce it?"
        ]
    },
    {
        "keywords": ["rejection", "rejected", "why reject", "declined", "reasons", "denied"],
        "reply": (
            "### ⚠️ Top Reasons for Loan Rejection and How to Fix Them\n\n"
            "Banks use automated underwriting algorithms to evaluate credit risk. The most common rejection triggers include:\n\n"
            "1. **Low Credit Score (< 650)**: High past default risk or insufficient credit history.\n"
            "2. **High Debt-to-Income (DTI) Ratio (> 50%)**: If more than 50% of your monthly income is already committed to existing EMIs, lenders fear repayment stress.\n"
            "3. **Unstable Employment History**: Job hopping or less than 1-2 years continuous experience in current employment/business.\n"
            "4. **Inconsistent Documentation**: Mismatches in salary slips, PAN/Aadhaar details, or unverified income.\n"
            "5. **High Loan Amount vs. Income**: Requesting an amount exceeding 4-5x your annual gross income without collateral or co-borrower.\n\n"
            "**💡 Actionable Next Steps:** Consider adding a co-applicant with a solid credit profile, opting for a longer tenure to reduce EMI, or reducing the requested loan amount."
        ),
        "suggestions": [
            "How can I improve my CIBIL score quickly?",
            "Should I choose a longer tenure or higher EMI?",
            "How does employment type affect interest rates?"
        ]
    },
    {
        "keywords": ["eligibility", "calculate eligibility", "qualify", "how much loan", "how do banks calculate", "dti", "foir"],
        "reply": (
            "### 📊 How Banks Calculate Your Loan Eligibility\n\n"
            "Lenders primarily use the **FOIR (Fixed Obligation to Income Ratio)** and **DTI (Debt-to-Income Ratio)** to determine maximum loan sanction:\n\n"
            "#### Standard Formula Used by Lenders:\n"
            "- **Max Allowable EMI** = `(Monthly Net Income × 50%) - Existing Monthly EMIs`\n"
            "- **Max Loan Amount** = Calculated based on Max EMI, prevailing interest rate, and preferred tenure.\n\n"
            "#### Key Eligibility Pillars:\n"
            "| Parameter | Ideal Benchmark | Impact |\n"
            "| :--- | :--- | :--- |\n"
            "| **CIBIL Score** | 750 or above | High - Lowest interest rate |\n"
            "| **FOIR / DTI** | Under 40% - 50% | Critical - Determines loan size |\n"
            "| **Employment** | Govt / Reputed MNC / 2+ yrs stable | Moderate - Better terms |\n"
            "| **Dependents** | Lower dependent ratio | Improves disposable income |"
        ),
        "suggestions": [
            "What are the best ways to reduce my existing EMI burden?",
            "What documents are needed for salaried vs self-employed?",
            "How to check loan approval probability on this platform?"
        ]
    },
    {
        "keywords": ["documents", "documentation", "papers", "kyc", "what do i need to apply", "required docs"],
        "reply": (
            "### 📑 Essential Documents Required for Loan Applications\n\n"
            "Having your documentation organized in advance ensures swift verification and prevents processing delays:\n\n"
            "#### 1. Identity & Address Proof (KYC):\n"
            "- PAN Card (mandatory for credit check & tax compliance)\n"
            "- Aadhaar Card / Passport / Voter ID / Driving License\n\n"
            "#### 2. Income Proof (Salaried Applicants):\n"
            "- Last 3 to 6 months salary slips\n"
            "- Last 6 months bank statement showing salary credits\n"
            "- Form 16 / ITR of the last 2 assessment years\n\n"
            "#### 3. Income Proof (Self-Employed / Business):\n"
            "- Last 2-3 years audited Balance Sheet & Profit/Loss account\n"
            "- Last 2-3 years ITR with computation of income\n"
            "- Last 12 months current bank account statement\n"
            "- Business registration/GST certificate\n\n"
            "#### 4. Property Documents (for Home/Mortgage Loans):\n"
            "- Sale deed, Title chain documents, Approved floor plan, NOC from builder/society."
        ),
        "suggestions": [
            "What is a good CIBIL score for personal vs home loans?",
            "How do banks evaluate self-employed loan applicants?",
            "How does loan tenure impact total interest paid?"
        ]
    },
    {
        "keywords": ["emi", "tenure", "calculate emi", "formula", "longer tenure", "shorter tenure", "prepayment", "foreclosure"],
        "reply": (
            "### ⚖️ Equated Monthly Installment (EMI) & Tenure Planning\n\n"
            "An **EMI** consists of both Principal and Interest components. In the initial years, the interest portion dominates; towards the end of tenure, principal repayment dominates.\n\n"
            "#### Mathematical EMI Formula:\n"
            "$$\\text{EMI} = \\frac{P \\times r \\times (1 + r)^n}{(1 + r)^n - 1}$$\n"
            "- **P** = Principal loan amount\n"
            "- **r** = Monthly interest rate (Annual rate / 12 / 100)\n"
            "- **n** = Tenure in number of months\n\n"
            "#### Shorter vs. Longer Tenure:\n"
            "- **Shorter Tenure (e.g. 3-5 yrs)**: Higher monthly EMI, but dramatically lower total interest paid.\n"
            "- **Longer Tenure (e.g. 15-20 yrs)**: Lower monthly EMI, easier approval under DTI rules, but higher total interest outlay.\n\n"
            "**💡 Smart Prepayment Strategy:** Paying just **one extra EMI each year** or increasing your EMI by 5% annually can reduce a 20-year home loan tenure down to ~12 years!"
        ),
        "suggestions": [
            "How does CIBIL score affect EMI?",
            "What is Debt-to-Income (DTI) ratio?",
            "Why do loans get rejected?"
        ]
    },
    {
        "keywords": ["hello", "hi", "hey", "who are you", "what can you do", "help"],
        "reply": (
            "### 👋 Hello! I'm your AI Loan & Financial Assistant\n\n"
            "I'm here to help you navigate borrowing, credit, and personal finance with clear, unbiased insights. Here's what you can ask me:\n\n"
            "- **Loan Eligibility & Approval**: Understand how banks evaluate income, employment, and DTI ratios.\n"
            "- **Credit Score (CIBIL/Experian)**: Actionable ways to boost your score to 750+ and fix reporting errors.\n"
            "- **Loan Types**: Personal loans, Home loans, Auto loans, Education loans, and Gold loans.\n"
            "- **EMI & Interest Rates**: Fixed vs. floating rates, EMI formulas, and smart prepayment strategies.\n"
            "- **Required Documentation**: Step-by-step paperwork checklist for quick approval.\n\n"
            "What financial question would you like to explore today?"
        ),
        "suggestions": [
            "How to boost CIBIL score to 750+?",
            "What is the difference between fixed and floating interest rates?",
            "What is a loan and how do banks calculate eligibility?"
        ]
    }
]

DEFAULT_SUGGESTIONS = [
    {
        "title": "Loan Eligibility Calculation",
        "prompt": "How do banks calculate my maximum loan eligibility and allowable EMI?",
        "category": "Loan Approval Queries",
        "icon": "calculator"
    },
    {
        "title": "Common Rejection Causes",
        "prompt": "What are the most common reasons for loan rejection and how can I avoid them?",
        "category": "Loan Approval Queries",
        "icon": "shield-alert"
    },
    {
        "title": "Boost CIBIL Score to 750+",
        "prompt": "What actionable steps can I take to improve my CIBIL score to 750 or higher?",
        "category": "Credit Score Queries",
        "icon": "trending-up"
    },
    {
        "title": "Shorter vs Longer Tenure",
        "prompt": "Should I choose a longer tenure with lower EMI or shorter tenure with higher EMI?",
        "category": "EMI & Planning",
        "icon": "clock"
    }
]


def _call_gemini_api(message: str, history: List[dict], context: Optional[dict] = None) -> Optional[str]:
    """
    Directly queries Google Gemini LLM via official Generative Language REST API.
    Supports GEMINI_API_KEY or GOOGLE_API_KEY environment variables.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "### ⚠️ Configuration Required\n\nTo use the AI Assistant, you must add your **GEMINI_API_KEY** to the `.env` file or environment variables in your deployment dashboard."

    # Supported fast model endpoints
    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-omni-flash-preview"
    ]

    # Inject Knowledge Base into System Instruction
    full_system = SYSTEM_INSTRUCTION + "\n\nPre-defined Knowledge Base (Always use this exact info when asked about these topics):\n"
    for kb in KNOWLEDGE_BASE:
        full_system += f"- {kb['reply']}\n"

    if context:
        full_system += f"\n\nApplicant Financial Context:\n{json.dumps(context, indent=2)}"

    # Merge consecutive messages with the same role to strictly alternate
    raw_contents = []
    for item in history[-6:]:
        role = "user" if item.get("role") == "user" else "model"
        raw_contents.append({"role": role, "text": item.get("content", "")})
    
    raw_contents.append({"role": "user", "text": message})

    contents = []
    for item in raw_contents:
        if contents and contents[-1]["role"] == item["role"]:
            contents[-1]["parts"][0]["text"] += "\n\n" + item["text"]
        else:
            contents.append({
                "role": item["role"],
                "parts": [{"text": item["text"]}]
            })

    payload = {
        "systemInstruction": {
            "parts": [{"text": full_system}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 1024,
        },
        "tools": [
            {"googleSearch": {}}
        ]
    }

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    resp_data = response.json()
                    candidates = resp_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                else:
                    print(f"Gemini API HTTP Error {response.status_code} for model {model_name}: {response.text}")
        except Exception as e:
            print(f"Gemini API Error for model {model_name}: {e}")
            continue

    return None





def process_chat_message(
    message: str,
    history: Optional[List[Any]] = None,
    context: Optional[dict] = None
) -> dict:
    """
    Main entry point for handling AI Loan Assistant chat queries.
    Tries Google Gemini first, falls back gracefully to expert financial engine.
    """
    history_list = []
    if history:
        for item in history:
            if hasattr(item, "model_dump"):
                history_list.append(item.model_dump())
            elif isinstance(item, dict):
                history_list.append(item)

    # Check if the message matches any pre-defined knowledge base topics first
    lower_message = message.lower().strip()
    for kb in KNOWLEDGE_BASE:
        for keyword in kb["keywords"]:
            if keyword in lower_message:
                return {
                    "reply": kb["reply"],
                    "suggestions": kb.get("suggestions", [
                        "What documents are required for quick approval?",
                        "How does my CIBIL score affect interest rates?",
                        "How can I lower my monthly EMI burden?"
                    ]),
                    "model": "local-knowledge-base",
                    "status": "success",
                }

    # Call Gemini API exclusively
    gemini_reply = _call_gemini_api(message, history_list, context)

    suggestions = [
        "What documents are required for quick approval?",
        "How does my CIBIL score affect interest rates?",
        "How can I lower my monthly EMI burden?"
    ]
    
    return {
        "reply": gemini_reply if gemini_reply else "### ❌ Error\n\nUnable to connect to the Gemini API. Please check your internet connection or API key.",
        "suggestions": suggestions,
        "model": "google-gemini",
        "status": "success",
    }


def get_curated_suggestions() -> dict:
    """
    Returns curated list of suggested prompt topics and categories.
    """
    categories = [
        "Loan Approval Queries",
        "Credit Score Queries",
        "EMI & Planning",
    ]
    return {
        "categories": categories,
        "suggestions": DEFAULT_SUGGESTIONS,
    }
