#!/bin/bash
#SBATCH --job-name=Medium_L_S
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00



export HF_HOME="/var/autofs/cl/home2/share/huggingface/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_TOKEN="hftoken"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1



# Capture timestamp for logging
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")



# Base Directories
BASEDIR=/Sinhala
WS=$BASEDIR
DATA=$WS/data
LOG_DIR=$WS/Log
OUTPUT=$WS/Results

# Set Difficulty Level from Input Argument
DIFFICULTY_LEVEL="medium" # easy, medium , hard
DATASET_PATH="$DATA/$DIFFICULTY_LEVEL"

# Set Fixed Few-Shot Value
FEW_SHOT=0  # Single value, no loop needed
LANGUAGE="sinhala"  # Change if needed english prompt


# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging Setup
EXPERIMENT_LOG="$LOG_DIR/exp_${DIFFICULTY_LEVEL}_log_$TIMESTAMP.txt"
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
   "Qwen/Qwen2.5-72B-Instruct"
    
)


# Iterate over JSON files in dataset
for JSON_FILE in "$DATASET_PATH"/*.json; do
    FILE_NAME=$(basename "$JSON_FILE")

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
            python3 /src/evaluate.py\
                --input "$JSON_FILE" \
                --output "$OUTPUT_DIR" \
                --model "$MODEL" \
                --few_shot "$FEW_SHOT" \
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
