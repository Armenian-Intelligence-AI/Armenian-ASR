from fastapi import FastAPI, UploadFile, File
from .asr_model import predict_asr
from .ner_model import run_ner_classifier
from .voice_activity_detection import run_voice_activity_detector
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/invocations")
async def invocations(file: UploadFile = File(...)):
    # Read the uploaded file
    audio_data = await file.read()
    audio_file = BytesIO(audio_data)
    
    # Check if the uploaded audio contains speech
    is_speech = run_voice_activity_detector(audio_file)
    if not is_speech:
        return {'prediction': '', 'duration': 0, 'is_speech': False}

    # Run the ASR prediction
    prediction_text, duration = predict_asr(audio_data)
    if prediction_text:
        prediction_text = run_ner_classifier(prediction_text)

    return {'prediction': prediction_text, 'duration': duration, 'is_speech': True}

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)