#!/bin/bash

# ==============================
# Usage:
# ./run_model.sh <model_name> <mode>
#
# mode:
#   single    -> only run single executor
#   dcipher   -> only run dcipher
#   both      -> run single then dcipher
# ==============================

MODEL="$1"
MODE="$2"

if [ -z "$MODEL" ] || [ -z "$MODE" ]; then
    echo "Usage: ./run_model.sh <model_name> <mode>"
    echo ""
    echo "Modes:"
    echo "  single"
    echo "  dcipher"
    echo "  both"
    echo ""
    echo "Available models:"
    echo "  llama3.1"
    echo "  qwen3-coder-next:cloud"
    echo "  cogito-2.1:671b-cloud"
    echo "  gpt-oss:120b-cloud"
    echo "  deepseek-v3.1:671b-cloud"
    echo "  gemma3:27b-cloud"
    exit 1
fi

CHALLENGES=(
"2023q-rev-baby_s_third"
"2023f-web-shreeramquest"
"2023q-for-1black0white"
"2018q-pwn-bigboy"
"2019q-cry-super_curve"
)

echo "============================================="
echo "Model: $MODEL"
echo "Mode : $MODE"
echo "============================================="

SINGLE_LOGDIR="logs_single_executor_${MODEL//[:.]/_}"
DCIPHER_LOGDIR="logs_dcipher_${MODEL//[:.]/_}"

mkdir -p $SINGLE_LOGDIR
mkdir -p $DCIPHER_LOGDIR

for CH in "${CHALLENGES[@]}"
do
    echo ""
    echo "---------------------------------------------"
    echo "Challenge: $CH"
    echo "---------------------------------------------"

    # ==============================
    # SINGLE EXECUTOR
    # ==============================
    if [ "$MODE" = "single" ] || [ "$MODE" = "both" ]; then
        echo "Running SINGLE EXECUTOR..."
        python3 run_single_executor.py \
            --split test \
            --challenge $CH \
            --enable-autoprompt \
            --executor-model $MODEL \
            --autoprompter-model $MODEL \
            --logdir $SINGLE_LOGDIR

        echo "SINGLE EXECUTOR completed."
    fi

    # ==============================
    # DCIPHER
    # ==============================
    if [ "$MODE" = "dcipher" ] || [ "$MODE" = "both" ]; then
        echo "Running DCIPHER..."
        python3 run_dcipher.py \
            --split test \
            --challenge $CH \
            --enable-autoprompt \
            --planner-model $MODEL \
            --executor-model $MODEL \
            --autoprompter-model $MODEL \
            --logdir $DCIPHER_LOGDIR

        echo "DCIPHER completed."
    fi

done

echo ""
echo "============================================="
echo "Completed for $MODEL in $MODE mode"
echo "============================================="