from pydantic import BaseModel, Field


class TouchupRequest(BaseModel):
    x: int
    y: int
    hex_color: str
    tolerance: int = Field(default=18, ge=1, le=100)
    feather: int = Field(default=3, ge=0, le=20)
