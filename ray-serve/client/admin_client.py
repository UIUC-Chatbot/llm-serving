import argparse
import pprint
import requests

parser = argparse.ArgumentParser(description="LLM Serving Admin Client")
parser.add_argument("--key", help="Admin Key", type=str, default="IloveRocknRoll")
parser.add_argument(
    "--mode",
    help="0: get model route; 1: delete model; 2: list models; 3: dump config; 4: reset",
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
    print("Listing all models")
    data = {
        "key": "IloveRocknRoll",
        "mode": "list",
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    response = response.json()
    print(f"\nModel Pool: {len(response['model_pool'])}")
    pprint.pprint(response["model_pool"])
    print(f"\nUnsupported Models: {len(response['model_unsupported'])}")
    pprint.pprint(response["model_unsupported"])

elif args.mode == 3:
    print("Dumping config to file")
    data = {
        "key": "IloveRocknRoll",
        "mode": "dump_config",
        "config_dump_path": args.config_dump_path,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)

elif args.mode == 4:
    print("Resetting LLM service")
    data = {
        "key": "IloveRocknRoll",
        "mode": "reset",
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)

else:
    print("Invalid mode")
