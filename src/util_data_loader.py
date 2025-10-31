import json
import os
import pandas as pd
from datasets import Dataset

def load_dataset(json_file_path):
    """Load dataset from JSON file into structured format."""
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    instructions = []
    for item in data:
        q_no = item.get("q_no", "Unknown_Q_No")
        subject = item.get("subject", "Unknown Subject")
        category = item.get("category", "Unknown Category")
        question = item.get("question", "Unknown Question")
        choices = item.get("choices", [])
        answer = item.get("answer", None)

        # Metadata fields
        metadata = item.get("metadata", {})
        subject_original = metadata.get("subject_original", "Unknown Subject Original")
        difficulty = metadata.get("difficulty", "Unknown Difficulty")
        grade = metadata.get("grade", "Unknown Grade")
        question_type = metadata.get("type", "Unknown Type") 
        source = metadata.get("source", "Unknown Source")

        instructions.append({
            "q_no": q_no,
            "subject": subject,
            "category": category,
            "question": question,
            "choices": choices,
            "answer": answer,
            "subject_original": subject_original,
            "difficulty": difficulty,
            "grade": grade,
            "type": question_type,  
            "source": source,
        })

    return Dataset.from_pandas(pd.DataFrame(instructions))

def load_json(file_name):
    """Load cached results from JSONL file."""
    data = {}
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            json_lines = f.readlines()
            for line in json_lines:
                try:
                    json_obj = json.loads(line.strip())
                    data[json_obj['prompt']] = json_obj['results']
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line in {file_name}")
    return data

def append_to_jsonl(file_name, data):
    """Append new results to JSONL file."""
    with open(file_name, "a", encoding="utf-8") as file:
        json_str = json.dumps(data, ensure_ascii=False)
        file.write(json_str + "\n")




def load_few_shot_examples(few_shot_path, num_examples):
    """Load few-shot examples from a dataset JSON file."""
    if num_examples == 0 or not os.path.exists(few_shot_path):
        print("Few-shot example file not found or not required. Using zero-shot mode.")
        return []

    with open(few_shot_path, "r", encoding="utf-8") as json_file:
        few_shot_data = json.load(json_file)

    # Extract only the first `num_examples` entries
    return [
        {
            "question": ex.get("question", "Unknown Question"),  # Extract question
            "choices": ex.get("choices", []),  # Extract choices
            "answer": ex.get("answer", "Unknown Answer")  # Extract answer
        }
        for ex in few_shot_data[:num_examples]
    ]

#gpt output
def create_output_row(entry, outputs, extracted_value):
    """
    Formats a single row for output CSV using the entry data and model output.
    """
    metadata = entry.get("metadata", {})
    return {
        "q_no": entry.get("q_no"),
        "subject": entry.get("subject"),
        "category": entry.get("category", "general"),
        "question": entry.get("question"),
        "choices": entry.get("choices"),
        "answer": entry.get("answer"),
        "subject_original": metadata.get("subject_original"),
        "difficulty": metadata.get("difficulty"),
        "grade": metadata.get("grade"),
        "type": metadata.get("type"),
        "source": metadata.get("source"),
        "model_predicted_answer": outputs,
        "generated_text": extracted_value

    }
