import argparse
import requests

parser = argparse.ArgumentParser(description="LLM-Serving Client Program")
parser.add_argument("--endpoint", help="LLM-Serving Endpoint", type=str)
parser.add_argument(
    "--mode",
    help="0: retrieve model route; 1: send prompt; 2: delete model",
    type=int,
)
parser.add_argument(
    "--model-name", help="Model Name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
)

args = parser.parse_args()
model_pool_endpoint = "127.0.0.1:8000/llm"
if args.mode == 0:
    print(f"Getting route for model {args.model_name}")
    data = {
        "mode": "get",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{model_pool_endpoint}", json=data)
    route = response.text
    print(route)
elif args.mode == 1:
    print(f"Sending prompt to model {args.model_name}")
    while True:
        data = input("Type something here ...\n")
        prompt = {
            "prompts": data,
        }
        response = requests.post(f"http://{args.endpoint}", json=prompt)
        output_text = response.text
        print(output_text)
elif args.mode == 2:
    print(f"Deleting model {args.model_name}")
    data = {
        "mode": "delete",
        "model_name": args.model_name,
    }
    response = requests.post(f"http://{model_pool_endpoint}", json=data)
    print(response.text)
