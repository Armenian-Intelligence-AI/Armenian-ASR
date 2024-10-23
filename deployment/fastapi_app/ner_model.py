import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def ner_model_fn():
    # Use the correct local path to the model directory
    model_path = '/opt/ml/model/ner_model'
    
    # Load the tokenizer and model from the local path
    ner_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

    return model, ner_tokenizer

ner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ner_model, ner_tokenizer = ner_model_fn()

ALLOWED_PUNCTUATIONS = {',', '։', '՝', '՞', '-', '.'}

PHRASES_WITH_I = [ "ի վեր", "ի վերջո", "ի նպաստ", "ի հեճուկս", "ի դեպ" "ի նշան", "ի պատիվ", "ի դեմ", "ի պաշտպանություն", "ի պահպանություն", "ի միջի", "ի հիշատակ", "ի ցույց", "ի գործ"]

# Function to replace 'եւ' with 'և'
def post_process_ner(sentence):
    sentence =  sentence.replace('եւ', 'և')
    return sentence if sentence.endswith('։') else sentence + '։'

def correct_sentence(input_sentence):
    # Tokenize input sentence
    tokenized_input = ner_tokenizer(
        input_sentence.split(),
        truncation=True,
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        max_length=128
    ).to(ner_model.device)

    # Get model predictions
    with torch.no_grad():
        output = ner_model(**tokenized_input)

    # Extract predicted token IDs and convert back to tokens
    predicted_ids = output.logits.argmax(dim=2)[0]
    tokens = ner_tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])

    # Define label names and special tokens
    label_names = ['O', '1', '2']
    special_tokens = set(ner_tokenizer.all_special_tokens)

    corrected_sentence = []
    current_word = ""

    # Process each token and its predicted label
    for token, predicted_id in zip(tokens, predicted_ids):
        label = label_names[predicted_id]

        if token in special_tokens:
            continue

        # Handle word continuation or start
        if token.startswith("▁"):
            if current_word:
                corrected_sentence.append(current_word)
            current_word = token[1:]
        else:
            current_word += token

        # Apply corrections based on label
        if label == '2':
            current_word = current_word.upper()
        elif label != 'O':
            current_word = current_word.capitalize()

    if current_word:
        corrected_sentence.append(current_word)

    # Join corrected words to form the final sentence
    final_sentence = " ".join(corrected_sentence)
    return final_sentence

# Function to add spaces around punctuations
def add_space_between_punctuation(sentence, punctuations):
    # Regular expression for allowed punctuations
    pattern = f"([{''.join(re.escape(p) for p in punctuations)}])"

    # Add space around hyphen
    corrected_sentence = re.sub(r"(\w)-(\w)", r"\1 - \2", sentence)

    # Add space before and after other punctuations
    corrected_sentence = re.sub(rf"(\S)({pattern})", r"\1 \2", corrected_sentence)
    corrected_sentence = re.sub(rf"({pattern})(\S)", r"\1 \2", corrected_sentence)

    return corrected_sentence


# Function to merge spaces around punctuations back to original form
def merge_same_punctuation(sentence, punctuations):
    # Regular expression for allowed punctuations
    pattern = f"([{''.join(re.escape(p) for p in punctuations)}])"

    # Merge spaces around hyphen
    merged_sentence = re.sub(r"\s-\s", "-", sentence)

    # Merge spaces around other punctuations
    merged_sentence = re.sub(rf"\s({pattern})", r"\1", merged_sentence)

    return merged_sentence

def add_hyphen(sentence):
    # Avoid changing the phrases in the valid list
    for phrase in PHRASES_WITH_I:
        if phrase in sentence:
            sentence = sentence.replace(phrase, phrase.replace(" ", "_"))  # Temporarily replace valid phrases

    # Add hyphen before standalone "ի" when it's not part of a valid phrase
    updated_sentence = re.sub(r'\b(\w+)\s(ի)\b', r'\1-\2', sentence)

    # Restore the valid phrases back to their original form
    for phrase in PHRASES_WITH_I:
        sentence_with_valid_phrases = phrase.replace(" ", "_")
        updated_sentence = updated_sentence.replace(sentence_with_valid_phrases, phrase)

    return updated_sentence

def run_ner_classifier(input_sentence):
    input_sentence = post_process_ner(input_sentence)
    corrected_sentence = add_space_between_punctuation(add_hyphen(input_sentence), ALLOWED_PUNCTUATIONS)
    after_model = correct_sentence(corrected_sentence)
    merged_sentence = merge_same_punctuation(post_process_ner(after_model), ALLOWED_PUNCTUATIONS)
    return merged_sentence
