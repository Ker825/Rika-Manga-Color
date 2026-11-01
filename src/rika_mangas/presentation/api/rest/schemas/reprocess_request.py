from pydantic import BaseModel, Field


class ReprocessRequest(BaseModel):
    style: str = "neutral"
    fidelity: float = Field(default=1.0, ge=0.0, le=1.0)
