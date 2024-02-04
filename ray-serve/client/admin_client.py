import argparse
import requests

parser = argparse.ArgumentParser(description="LLM Serving Admin Client")
parser.add_argument("--key", help="Admin Key", type=str, default="IloveRocknRoll")
parser.add_argument(
    "--mode",
    help="0: get model route; 1: delete model; 2: dump config",
    type=int,
    required=True,
)
parser.add_argument(
    "--endpoint",
    help="LLM-Serving Endpoint",
    type=str,
    default="0.0.0.0:5004/llm/admin",
)
parser.add_argument(
    "--model-name", help="Model Name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
)
parser.add_argument("--model-type", help="Model Type", type=str, default="vllm_raw")
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--config-dump-path", type=str, default="latest_config.yaml")
args = parser.parse_args()


if args.mode == 0:
    print(f"Requesting route for model {args.model_name}")
    data = {
        "key": "IloveRocknRoll",
        "mode": "get",
        "model_name": args.model_name,
        "model_type": args.model_type,
        "gpus_per_replica": args.num_gpus,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    output_text = response.text
    print(output_text)
elif args.mode == 1:
    print(f"Deleting model {args.model_name}")
    data = {
        "key": "IloveRocknRoll",
        "mode": "delete",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)
elif args.mode == 2:
    print("Dumping config to file")
    data = {
        "key": "IloveRocknRoll",
        "mode": "dump_config",
        "model_name": args.model_name,
        "config_dump_path": args.config_dump_path,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)
else:
    print("Invalid mode")
