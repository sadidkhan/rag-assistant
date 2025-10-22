from pydantic import BaseModel
from typing import List

class ChatMessageDTO(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessageDTO] = []

class ChatResponse(BaseModel):
    reply: str
