from transformers import (WhisperFeatureExtractor, WhisperTokenizer,
                          WhisperForConditionalGeneration, WhisperProcessor, 
                          Seq2SeqTrainingArguments, TrainerCallback, Seq2SeqTrainer)
from datasets import Dataset, DatasetDict, Audio, Features, Value, load_from_disk
from huggingface_hub import notebook_login, HfApi
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import subprocess
import os
from torch.utils.tensorboard import SummaryWriter

try:
    # Run the nvidia-smi command and capture the output
    gpu_info = subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.STDOUT, text=True)
    if 'failed' in gpu_info:
        print('Not connected to a GPU')
    else:
        print(gpu_info)
except subprocess.CalledProcessError as e:
    # Handle the case where nvidia-smi is not found or another error occurs
    print(f"Error occurred while checking GPU: {e.output}")

huggingface_token = ''
os.environ['HF_TOKEN'] = huggingface_token

destination_path = "dataset"

loaded_dataset = load_from_disk(destination_path)
print("Loaded Dataset:", loaded_dataset)


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="armenian", task="transcribe")

common_voice = loaded_dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    # Remove the sentence and audio fields as they are no longer needed
    batch.pop("sentence", None)
    batch.pop("audio", None)
    
    return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=6)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.generation_config.language = "armenian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Count the number of parameters
total_params = sum(p.numel() for p in model.parameters())

print(f'Total number of parameters: {total_params}')

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="armenian", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-armenian",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=3000,
    max_steps=400000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=40000,
    eval_steps=40000,
    logging_steps=500,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=16,
    push_to_hub=False,
)

writer = SummaryWriter(log_dir="./whisper-small-armenian-logs")


class CustomCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                writer.add_scalar(key, value, state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Check if the current epoch is a multiple of 5 or the last epoch
        # if state.epoch % 2 == 0:
        if True:
            # Select 10 random items from the validation set
            eval_dataset = common_voice["test"].shuffle(seed=42).select(range(25))
            # Evaluate the model on these items
            outputs = trainer.predict(eval_dataset)
            # Decode predictions and references
            pred_str = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)
            label_ids = outputs.label_ids
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            # Prepare the results to be written to the file
            results = []
            for i in range(10):
                results.append(f"PREDICTED: {pred_str[i]}")
                results.append(f"REFERENCE: {label_str[i]}")
                results.append("-" * 30)

            # Write the results to output.txt, overwriting the existing content
            with open(f'epoch_logs/output_{state.epoch}.txt', 'w') as file:
                for line in results:
                    file.write(line + '\n')

            for i, (pred, ref) in enumerate(zip(pred_str, label_str)):
                writer.add_text(f"Prediction/Example_{i}", f"Predicted: {pred}\nReference: {ref}", state.global_step)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[CustomCallback()]
)
processor.save_pretrained(training_args.output_dir)

trainer.train()

output_dir = "./whisper_armenian_fine_tuned"

# Save the model, tokenizer, and processor
trainer.model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

writer.close()
