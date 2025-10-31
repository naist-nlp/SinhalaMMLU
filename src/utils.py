import torch
import torch.nn.functional as F
import numpy as np

def softmax(x):
    """Compute softmax for a given array."""
    z = x - np.max(x)  # Stability trick to prevent overflow
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

@torch.no_grad()
def predict_classification_causal_by_number(model, tokenizer, input_text, choices, device):
    """
    Predicts an answer (1, 2, 3, 4, or 5) based on the probability of the first generated token.
    """
    torch.manual_seed(42)

    # Use the choices from the dataset
    choice_labels = [str(i + 1) for i in range(len(choices))]  # ['1', '2', '3', '4', '5']
    
    # Get token IDs for the dynamic choices
    choice_ids = [tokenizer.encode(label, add_special_tokens=False)[-1] for label in choice_labels]

    # Tokenize input prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    #print(f"Tokenized Input IDs: {inputs['input_ids']}")

    # Handle Falcon models by removing `token_type_ids`
    if hasattr(model.config, "model_type") and model.config.model_type == 'falcon':
        #print("Falcon model detected. Removing `token_type_ids`.")
        inputs.pop("token_type_ids", None)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Get model outputs
    outputs = model(**inputs)
    # Convert logits from BFloat16 → Float32 for safe indexing
    first_token_logits = outputs.logits[:, -1, :].to(torch.float32)
    #Convert choice indices to tensor for indexing
    choice_tensor = torch.tensor(choice_ids, dtype=torch.long, device=first_token_logits.device)
    #Extract logits for the dynamic answer choices
    choice_logits = first_token_logits[:, choice_tensor].detach().cpu().numpy()
    #print(f"Model Logits for Choices: {choice_logits[0]}")
    # Compute softmax probabilities
    conf = softmax(choice_logits[0])
    #print(f"Softmax Probabilities: {conf}")
    # Select the answer with the highest probability
    pred = choice_labels[np.argmax(choice_logits[0])]


#Why extract the last token's logits?
#Since the model generates output token-by-token, the last generated token often contains the model’s prediction for the answer.
#These logits represent probabilities (before applying softmax) for each possible token in the vocabulary.

    # Debugging: Show final predicted answer
    #print(f"Predicted Answer: {pred} (with probability {conf[np.argmax(choice_logits[0])]:.4f})")

    return conf, pred


def extract_number_from_generated_answer(text):
    """
    Extracts the number (1-5) from the generated answer.
    If no number is found, it prints the original text and returns it.
    """
    pattern = r'(?:Answer:|පිළිතුර|සාවද්‍ය ප්‍රකාශය:)?\s*(\d)[\.\s]'  

    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Extract only the number

    print(f"⚠️ No number found, returning full text: {text}")
    return text
