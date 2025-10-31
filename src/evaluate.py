import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
import torch
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import logging
import numpy as np
from utils import predict_classification_causal_by_number  # Import utils function
from util_data_loader import load_dataset, load_json, append_to_jsonl, load_few_shot_examples  # Import from utils.py
from util_prompt import generate_instruction_prompt  # Import from templates.py

# Logging setup
logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)
set_seed(0)
hf_token = os.getenv("HUGGINGFACE_TOKEN")

IS_INSTRUCTION_MODEL = True


def main(file_path, output_dir, model_name, few_shot, few_shot_path, language, difficulty, intro):
    """Main function to run model inference with new dataset fields."""
    
    dataset = load_dataset(file_path)
    print(dataset.to_pandas())

    # Extract dataset metadata
    category = dataset["category"][0] if "category" in dataset.column_names else "unknown"
    question_type = dataset["type"][0] if "type" in dataset.column_names else "unknown"
    subject = dataset["subject"][0] if "subject" in dataset.column_names else "unknown"
    subject_original = dataset["subject_original"][0] if "subject_original" in dataset.column_names else "unknown"


   
    # Define cache directory inside the output directory
    cache_dir = os.path.join(output_dir, f'cache/{model_name}/{difficulty}')
    os.makedirs(cache_dir, exist_ok=True)
    file_name_cache = os.path.join(cache_dir, f"results_{subject}.jsonl")

    # Load cache if exists
    cache = load_json(file_name_cache) if os.path.exists(file_name_cache) else {}

    # Load few-shot examples
    few_shot_examples = load_few_shot_examples(few_shot_path, few_shot) if few_shot > 0 else []

    # Set device & load model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, 
                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, 
                                                 trust_remote_code=True, quantization_config=qconfig, 
                                                 low_cpu_mem_usage=True, use_cache=True, use_auth_token=hf_token)

    # Apply tokenization & template formatting
    def apply_template(example):
        """Format prompt with new dataset fields."""
        instruction = generate_instruction_prompt(
            level=difficulty,
            subject=subject,
            subject_original=subject_original,
            question=example['question'],
            choices=example['choices'],
            language=language,
            few_shot_examples=few_shot_examples,
            intro_type=intro
        )

        print(instruction)

        messages = [
            {"role": "system", "content": "You must output the Answer"},
            {"role": "user", "content": instruction}
        ]
        sentence = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": sentence}

    dataset = dataset.map(apply_template, batched=False)

    def calculate(examples):
        prompt = examples['prompt']
        if prompt in cache:
            result = cache[prompt]
            generated_text = result["results"]
            confidence = result["Confidence"]
        else:
            logits, generated_text = predict_classification_causal_by_number(model, tokenizer, prompt, examples["choices"], device)
            max_confidence = float(np.max(logits))
            confidence = max_confidence

            json_result = {
            "results": generated_text,
            "prompt": prompt,
            "Confidence": confidence
            }
            append_to_jsonl(file_name_cache, json_result)
            cache[prompt] = json_result

            #print(f"Model Prediction: {generated_text} | Confidence: {confidence}")

        return {"generated_text": generated_text, "confidence": confidence}


    dataset = dataset.map(calculate, batched=False)
    df = dataset.to_pandas()  
    print(df)

   
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output CSV file
    output_file = os.path.join(output_dir, f"{subject}_{model_name.replace('/', '_')}_{difficulty}_{category}_{language}.csv")

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Results saved to {output_file}")

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to save output JSON file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples (0-3)")
    parser.add_argument("--few_shot_path", type=str, default="few_shot_examples.json", help="Path to few-shot examples JSON file")
    parser.add_argument("--language", type=str, choices=["sinhala", "english"], default="sinhala", help="Language of instructions")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"], required=True, help="Difficulty level")
    parser.add_argument("--intro", type=str, default="True", help="Include subject introduction in prompt (True/False)")


    args = parser.parse_args()
    args.intro = args.intro.lower() == "true"
    
    main(args.input, args.output, args.model, args.few_shot, args.few_shot_path, args.language, args.difficulty, args.intro)
