# api-service/app/schemas.py

from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str