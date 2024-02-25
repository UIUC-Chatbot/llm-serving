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

if  [[ $# -ne 3 ]]; then
    echo "Accepts 3 arguments: <num_users> <model_name> <endpoint>"
    exit 1
fi

NUM_USERS=$1
echo "Number of concurrent users: $NUM_USERS"

MODEL_NAME=$2
echo "Model name: $MODEL_NAME"

ENDPOINT=$3
echo "Endpoint: $ENDPOINT"

sleep 3

for i in $(seq 1 ${NUM_USERS}); do
   python openai_client.py -e ${ENDPOINT} --model-name ${MODEL_NAME} -l -s &
done

wait