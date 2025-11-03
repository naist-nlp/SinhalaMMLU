#!/bin/bash
#SBATCH --job-name=easy_s
#SBATCH --output=llama%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00


export HF_HOME=""
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_TOKEN=""

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1



# Capture timestamp for logging
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")



# Base Directories
BASEDIR=Sinhala
WS=$BASEDIR
DATA=$WS/data
LOG_DIR=$WS/Log
OUTPUT=$WS/Results


#MODEL=$WS/model


# Set Difficulty Level from Input Argument
DIFFICULTY_LEVEL="easy"  # easy, medium, hard  levels
DATASET_PATH="$DATA/$DIFFICULTY_LEVEL"

#When using fewshot examples
FEW_SHOT_DATA=$WS/data/fewshot  # Path to few-shot examples (organized by difficulty)
FEW_SHOT_PATH="$FEW_SHOT_DATA/$DIFFICULTY_LEVEL"


# Set Fixed Few-Shot Value
FEW_SHOT=3  # Single value, no loop needed
LANGUAGE="sinhala"  # Change if needed english


# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging Setup
EXPERIMENT_LOG="$LOG_DIR/experiment_${DIFFICULTY_LEVEL}_${FEW_SHOT}log_$TIMESTAMP.txt"
exec > >(tee -a "$EXPERIMENT_LOG") 2>&1

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Experiment started at $(date)"
echo "Difficulty Level: $DIFFICULTY_LEVEL"
echo "Language: $LANGUAGE"
echo "Few-Shot Level: $FEW_SHOT"
echo "=========================================="





MODELS=(
    
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-32B"
    "CohereForAI/aya-expanse-32b" 

)



# Iterate over JSON files in dataset
for JSON_FILE in "$DATASET_PATH"/*.json; do
    FILE_NAME=$(basename "$JSON_FILE")

    BASE_NAME="${FILE_NAME%.json}"  # Remove .json extension
    BASE_NAME_STRIPPED=$(echo "$BASE_NAME" | sed 's/_f_[0-9]\+$//')

    FEW_SHOT_FILE="$FEW_SHOT_PATH/${BASE_NAME_STRIPPED}_fewshot.json"  # Construct few-shot file path

    # Skip dataset file if no matching few-shot file exists
    if [ ! -f "$FEW_SHOT_FILE" ]; then
        echo "Looking for few-shot file: $FEW_SHOT_FILE"

        echo "Skipping $FILE_NAME: Few-shot file not found."
        continue
    fi

    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL")

        for INTRO in "True" "False"; do
            # Define structured output directory: Model > Difficulty > Language > Few-shot > Intro
            INTRO_DIR="intro_${INTRO,,}"  # Converts "True" to "intro_true" and "False" to "intro_false"
            OUTPUT_DIR="$OUTPUT/$MODEL_NAME/$DIFFICULTY_LEVEL/$LANGUAGE/fewshot_$FEW_SHOT/$INTRO_DIR"
            mkdir -p "$OUTPUT_DIR"

            echo "Running: Model=$MODEL | File=$FILE_NAME | Difficulty=$DIFFICULTY_LEVEL | Language=$LANGUAGE | Few-Shot=$FEW_SHOT | Intro=$INTRO"
            echo "Saving results in: $OUTPUT_DIR"

            # Execute the main Python script with --intro True or --intro False
            python3 src/evaluate.py \
                --input "$JSON_FILE" \
                --output "$OUTPUT_DIR" \
                --model "$MODEL" \
                --few_shot "$FEW_SHOT" \
                --few_shot_path "$FEW_SHOT_FILE" \
                --language "$LANGUAGE" \
                --difficulty "$(echo $DIFFICULTY_LEVEL | tr '[:upper:]' '[:lower:]')" \
                --intro $INTRO

            echo "--------------------------------------------------------"
        done
    done
done

echo "=========================================="
echo "Experiment for $DIFFICULTY_LEVEL finished at $(date)"
echo "=========================================="
