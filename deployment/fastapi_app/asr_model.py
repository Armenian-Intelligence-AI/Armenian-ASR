
import os
import base64
import librosa
from io import BytesIO
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment

def stt_model_fn():
    model_path = os.path.join('/opt/ml/model', 'whisper_arm_stt')
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(asr_device)
    processor = WhisperProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor

asr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stt_model, stt_processor = stt_model_fn()

def check_audio_quiet_threshold(audio_bytes_io, quiet_threshold=-47):
    # Load the audio file from BytesIO object
    audio = AudioSegment.from_file(audio_bytes_io)

    # Calculate the dBFS (decibels relative to full scale) of the entire audio
    audio_dBFS = audio.dBFS
    print(f"Audio dBFS: {audio_dBFS:.2f} dBFS")  # Optional: For debugging purposes

    # Check against the quiet threshold
    if audio_dBFS > quiet_threshold:
        return True
    else:
        return False
    
def predict_asr(audio_data: bytes):
    audio_buffer = BytesIO(audio_data)
    if not check_audio_quiet_threshold(audio_buffer):
        return '', 0
    audio_buffer.seek(0)
    audio_input, sample_rate = librosa.load(audio_buffer, sr=16000)
    duration = librosa.get_duration(y=audio_input, sr=sample_rate)

    audio_input = torch.tensor(audio_input)
    inputs = stt_processor(audio_input, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(asr_device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = stt_model.generate(inputs["input_features"])

    predicted_ids = predicted_ids
    transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0], duration

def get_sampling_rate(audio_bytes):
    # Load the audio file from bytes using soundfile
    audio_data, sample_rate = sf.read(BytesIO(audio_bytes))
    return audio_data, sample_rate

