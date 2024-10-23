from pydantic import BaseModel

class PredictionRequest(BaseModel):
    audio: str