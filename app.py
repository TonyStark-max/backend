from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

try:
    from dotenv import load_dotenv
    base_dir = Path(__file__).parent
    load_dotenv(base_dir / ".env")
    load_dotenv(base_dir / ".env.example")
    load_dotenv(base_dir.parent / ".env")
except ImportError:
    pass

from routes.loan_routes import router
from routes.ntc_routes import router as ntc_router
from routes.assistant_routes import router as assistant_router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML API for 7-feature loan approval prediction and AI Loan Assistant",
    version="2.1.0",
)


# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [
    # Local development ports
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:3000",
    "http://127.0.0.1:3000",

    # Production Vercel deployment
    "https://loan-approval-prediction-xi-self.vercel.app",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,

    # Allow local development ports and Vercel preview deployments
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app",

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def read_root():
    return {"message": "Loan Approval Prediction API is running. Visit /docs for the API documentation."}

# Existing loan prediction + maximum eligible loan endpoints
app.include_router(router)

# Existing New-to-Credit functionality
app.include_router(ntc_router)

# New AI Loan Assistant
app.include_router(assistant_router)
