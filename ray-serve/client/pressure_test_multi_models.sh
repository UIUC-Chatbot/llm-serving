#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "Terminating background processes..."
    kill -- -$$
    wait
    echo "All processes terminated."
    exit
}

# Trap Ctrl-C (SIGINT) and call cleanup function
trap cleanup SIGINT

models=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "Qwen/Qwen-7B-Chat"
    "gpt2-xl"
    "baichuan-inc/Baichuan-7B"
    "EleutherAI/gpt-j-6b"
    "EleutherAI/pythia-12b"
    "facebook/opt-13b"
    "THUDM/chatglm2-6b"
    "THUDM/chatglm3-6b"
)
NUM_TOTAL_MODELS=${#models[@]}

if  [[ $# -ne 4 ]]; then
    echo "Accepts 4 arguments: <num_models> <num_users_per_model> <endpoint> <hf_key>"
    exit 1
fi

if [[ $1 -gt ${NUM_TOTAL_MODELS} ]]; then
    echo "Number of models should be less than or equal to ${NUM_TOTAL_MODELS}"
    exit 1
fi

NUM_MODELS_TO_TEST=$1
echo "Number of models to test: $NUM_MODELS_TO_TEST"

NUM_USERS_PER_MODEL=$2
echo "Number of users per model: $NUM_USERS_PER_MODEL"

ENDPOINT=$3
echo "Endpoint: $ENDPOINT"

HF_KEY=$4
echo "HF key: $HF_KEY"

echo ""

sleep 3

# Array to hold PIDs
declare -a PIDS

for i in $(seq 0 $((${NUM_MODELS_TO_TEST} - 1))); do
    model=${models[${i}]}
    ./pressure_test_single_model.sh ${NUM_USERS_PER_MODEL} ${model} ${ENDPOINT} ${HF_KEY} &
   PIDS[${i}]=$!
done

wait