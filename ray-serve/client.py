import argparse
import requests

parser = argparse.ArgumentParser(description="LLM-Serving Client Program")
parser.add_argument(
    "--endpoint", help="LLM-Serving Endpoint", type=str, default="127.0.0.1:8000/llm"
)
parser.add_argument(
    "--mode",
    help="0: get model route; 1: send prompt; 2: delete model; 3: dump config",
    type=int,
    required=True,
)
parser.add_argument(
    "--model-name", help="Model Name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
)

args = parser.parse_args()
if args.mode == 0:
    print(f"Requesting route for model {args.model_name}")
    data = {
        "mode": "get",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    output_text = response.text
    print(output_text)
elif args.mode == 1:
    print(f"Sending prompt to model {args.model_name}")
    while True:
        data = input("Type something here ...\n")
        prompt = {"prompt": data, "load_required": True}
        response = requests.post(f"http://{args.endpoint}", json=prompt)
        output_text = response.text
        print(output_text)
elif args.mode == 2:
    print(f"Deleting model {args.model_name}")
    data = {
        "mode": "delete",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{args.endpoint}", json=data)
    print(response.text)
elif args.mode == 3:
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
