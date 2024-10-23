import os
import pandas as pd
import random
from pydub import AudioSegment
import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Value, Audio

def preprocess_audio(file_path):
    try:
        # Try to load the audio file using soundfile
        audio, sample_rate = sf.read(file_path)
    except RuntimeError:
        # If loading fails, preprocess the audio file using pydub
        audio = AudioSegment.from_file(file_path)
        audio.export(file_path, format="wav")
        audio, sample_rate = sf.read(file_path)

    return file_path, sample_rate

def load_audio_text_pairs(folder_path, metadata_file):
    # Load the metadata CSV file
    metadata_df = pd.read_csv(metadata_file)

    audio_text_pairs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            base_name = os.path.splitext(file_name)[0]  # Keep as string
            audio_path = os.path.join(folder_path, file_name)

            # Check for the text row using the string base_name
            text_row = metadata_df[metadata_df.get('file_name', False) == base_name]

            text = text_row['normalized_transcription'].values[0] if text_row['normalized_transcription'].size > 0 else ''
            audio_text_pairs.append({"audio": audio_path, "sentence": text})

    return audio_text_pairs

def create_dataset_dict(folder_path, metadata_file='metadata.csv', train_size=0.9):
    audio_text_pairs = load_audio_text_pairs(folder_path + "/wavs", metadata_file)

    # Shuffle the data
    random.shuffle(audio_text_pairs)

    # Split the data into train and test sets
    train_size = int(len(audio_text_pairs) * train_size)
    train_pairs = audio_text_pairs[:train_size]
    test_pairs = audio_text_pairs[train_size:]

    # Convert to datasets
    train_data_dict = {
        "audio": [pair["audio"] for pair in train_pairs],
        "sentence": [pair["sentence"] for pair in train_pairs],
    }

    test_data_dict = {
        "audio": [pair["audio"] for pair in test_pairs],
        "sentence": [pair["sentence"] for pair in test_pairs],
    }

    features = Features({
        "audio": Audio(sampling_rate=16000),
        "sentence": Value("string"),
    })

    train_dataset = Dataset.from_dict(train_data_dict, features=features)
    test_dataset = Dataset.from_dict(test_data_dict, features=features)

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

dataset = create_dataset_dict('/path/to/data')
dataset.save_to_disk('dataset')
