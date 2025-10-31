import os
import pandas as pd
from collections import defaultdict

# Define paths and configuration
BASE_DIR = "Sinhala/Results"
OUTPUT_DIR = "Sinhala/overall_category_accuracy"
LANGUAGE = "sinhala"
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
KNOWN_CATEGORIES = {"humanities", "social_science", "stem", "language", "other", "Business studies"}
INTRO_VARIANTS = ["intro_true"] 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_accuracy(df):
    # Convert answer and generated_text to numeric (int), safely
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce").astype("Int64")
    df["generated_text"] = pd.to_numeric(df["generated_text"], errors="coerce").astype("Int64")

    # Compare for exact match
    correct = (df["answer"] == df["generated_text"]).sum()
    total = len(df)
    return correct, total

# Dictionary to store results: model -> category -> [correct_count, total_count]
results = defaultdict(lambda: defaultdict(lambda: [0, 0]))

# Iterate over difficulty levels
for DIFFICULTY in DIFFICULTY_LEVELS:
    print(f"\nProcessing difficulty: {DIFFICULTY}")

    # Use a sample model folder to discover available few-shot configurations
    sample_model_dir = os.path.join(BASE_DIR, sorted(os.listdir(BASE_DIR))[0], DIFFICULTY, LANGUAGE)
    fewshot_folders = [f for f in sorted(os.listdir(sample_model_dir)) if os.path.isdir(os.path.join(sample_model_dir, f))]

    # Iterate through few-shot prompt variations
    for fewshot in fewshot_folders:
        for model_name in sorted(os.listdir(BASE_DIR)):
            model_path = os.path.join(BASE_DIR, model_name, DIFFICULTY, LANGUAGE, fewshot)
            if not os.path.isdir(model_path):
                continue  # Skip if path doesn't exist

            for intro in INTRO_VARIANTS:
                intro_path = os.path.join(model_path, intro)
                if not os.path.exists(intro_path):
                    continue  # Skip if intro variant folder doesn't exist

                # Process each CSV file in the directory
                for file in sorted(os.listdir(intro_path)):
                    if not file.endswith(".csv"):
                        continue

                    filepath = os.path.join(intro_path, file)
                    try:
                        df = pd.read_csv(filepath)

                        # Ensure required columns are present
                        if all(col in df.columns for col in ['answer', 'generated_text', 'subject', 'category']):
                            category = str(df['category'].iloc[0]).strip()
                            subject = str(df['subject'].iloc[0]).strip()

                            # Skip unknown categories
                            if category not in KNOWN_CATEGORIES:
                                continue

                            # Compute accuracy
                            correct, total = compute_accuracy(df)

                            results[model_name][category][0] += correct
                            results[model_name][category][1] += total

                    except Exception as e:
                        print(f"Error reading {file} in {intro_path}: {e}")

# Convert aggregated results to final output format
records = []
for model, cat_data in results.items():
    total_correct = 0
    total_questions = 0
    row = {"model": model}
    for cat in sorted(KNOWN_CATEGORIES):
        correct, total = cat_data[cat]
        acc = correct / total if total > 0 else 0.0
        row[cat] = acc
        total_correct += correct
        total_questions += total
    row["average"] = total_correct / total_questions if total_questions > 0 else 0.0
    records.append(row)

# Save the final result as CSV and JSON
final_df = pd.DataFrame(records)
final_df.to_csv(os.path.join(OUTPUT_DIR, "final_category_accuracy.csv"), index=False)
#final_df.to_json(os.path.join(OUTPUT_DIR, "final_category_accuracy.json"), orient="records", indent=2)

print(f"Saved overall per-category accuracy across difficulties to {OUTPUT_DIR}")
