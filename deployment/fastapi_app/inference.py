from fastapi import FastAPI
from .schemas import PredictionRequest
from .asr_model import predict_asr
from .ner_model import run_ner_classifier
from .audio_classification import run_audio_classifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/invocations")
async def invocations(request: PredictionRequest):
    audio_base64 = request.audio
    is_speech = run_audio_classifier(audio_base64)
    if not is_speech:
        return {'prediction': '', 'duration': 0, 'is_speech': False}
    prediction_text, duration = predict_asr(audio_base64)
    if prediction_text:
        prediction_text = run_ner_classifier(prediction_text)
    return {'prediction': prediction_text, 'duration': duration, 'is_speech': True}

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    