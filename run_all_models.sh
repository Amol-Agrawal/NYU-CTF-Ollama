#!/bin/bash

MODELS=(
"qwen3-coder-next:cloud"
"cogito-2.1:671b-cloud"
"gpt-oss:120b-cloud"
"deepseek-v3.1:671b-cloud"
"gemma3:27b-cloud"
)

CHALLENGES=(
"2023q-rev-baby_s_third"
"2023f-web-shreeramquest"
"2023q-for-1black0white"
"2018q-pwn-bigboy"
"2019q-cry-super_curve"
)

for MODEL in "${MODELS[@]}"
do
    echo "#################################################"
    echo "Running for MODEL: $MODEL"
    echo "#################################################"

    # Create separate log directories per model
    SINGLE_LOGDIR="logs_single_executor_${MODEL//[:.]/_}"
    DCIPHER_LOGDIR="logs_dcipher_${MODEL//[:.]/_}"

    mkdir -p $SINGLE_LOGDIR
    mkdir -p $DCIPHER_LOGDIR

    for CH in "${CHALLENGES[@]}"
    do
        echo "---------------------------------------------"
        echo "Challenge: $CH"
        echo "---------------------------------------------"

        echo "Running SINGLE EXECUTOR..."
        python3 run_single_executor.py \
            --split test \
            --challenge $CH \
            --enable-autoprompt \
            --executor-model $MODEL \
            --autoprompter-model $MODEL \
            --logdir $SINGLE_LOGDIR

        echo "Running DCIPHER..."
        python3 run_dcipher.py \
            --split test \
            --challenge $CH \
            --enable-autoprompt \
            --planner-model $MODEL \
            --executor-model $MODEL \
            --autoprompter-model $MODEL \
            --logdir $DCIPHER_LOGDIR

        echo "Finished $CH for $MODEL"
        echo ""
    done

    echo "Completed all challenges for $MODEL"
    echo ""
done

echo "=============================================="
echo "ALL MODELS + ALL CHALLENGES COMPLETED"
echo "=============================================="
