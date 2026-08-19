from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="Role of the sender: 'user' or 'assistant' or 'system'",
    )
    content: str = Field(
        ...,
        description="Content of the message",
    )
    timestamp: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The user's query or prompt",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Previous messages in the conversation",
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional loan application context or financial parameters",
    )


class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str] = Field(
        default=[],
        description="Suggested follow-up queries or quick topics",
    )
    model: str = "google-gemini"
    status: str = "success"


class SuggestedPrompt(BaseModel):
    title: str
    prompt: str
    category: str
    icon: Optional[str] = None


class SuggestionsResponse(BaseModel):
    categories: List[str]
    suggestions: List[SuggestedPrompt]
