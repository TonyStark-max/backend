from fastapi import APIRouter
from schemas.assistant_schema import (
    ChatRequest,
    ChatResponse,
    SuggestionsResponse,
)
from services.assistant_service import (
    process_chat_message,
    get_curated_suggestions,
)

router = APIRouter(prefix="/assistant", tags=["AI Loan Assistant"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):
    """
    Endpoint for asking questions to the AI Loan Assistant.
    Supports chat history, applicant context, and returns suggestions.
    """
    result = process_chat_message(
        message=request.message,
        history=request.history,
        context=request.context,
    )
    return result


@router.get("/suggestions", response_model=SuggestionsResponse)
def get_prompt_suggestions():
    """
    Endpoint for getting curated prompt recommendations categorized by
    Loan Approval Queries, Credit Score Queries, and EMI & Planning.
    """
    return get_curated_suggestions()
