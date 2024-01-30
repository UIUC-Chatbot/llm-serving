import argparse
import requests

parser = argparse.ArgumentParser(description="Get Model Route")
parser.add_argument(
    "--mode",
    help="0: get model route; 1: delete model; 2: dump config",
    type=int,
    required=True,
)
parser.add_argument(
    "--endpoint", help="LLM-Serving Endpoint", type=str, default="127.0.0.1:8000/llm"
)
parser.add_argument("--model-name", help="Model Name", type=str, required=True)
parser.add_argument("--model-type", help="Model Type", type=str, required=True)
args = parser.parse_args()


if args.mode == 0:
    print(f"Requesting route for model {args.model_name}")
    data = {
        "mode": "get",
        "model_name": args.model_name,  # e.g., meta-llama/Llama-2-7b-chat-hf
        "model_type": args.model_type,  # e.g., vllm_openai
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    output_text = response.text
    print(output_text)
elif args.mode == 1:
    print(f"Deleting model {args.model_name}")
    data = {
        "mode": "delete",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)
elif args.mode == 2:
    print("Dumping config to file")
    data = {
        "mode": "dump_config",
        "model_name": args.model_name,
        "file_path": "latest_config.yaml",
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)
else:
    print("Invalid mode")
