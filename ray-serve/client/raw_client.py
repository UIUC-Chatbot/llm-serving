import argparse
import requests

parser = argparse.ArgumentParser(description="LLM-Serving Client Program")
parser.add_argument("--endpoint", help="LLM-Serving Endpoint", type=str, required=True)
parser.add_argument("--load-required", type=bool, default=False)
parser.add_argument("--loop", type=bool, default=False)
args = parser.parse_args()
if "." not in args.endpoint:
    args.endpoint = f"127.0.0.1:8000/{args.endpoint}"

print(f"Sending prompt to endpoint {args.endpoint}")
while True:
    if args.loop:
        input_data = "Can you tell me a joke?"
    else:
        input_data = input("Type something here ...\n")
    data = {"prompt": input_data, "load_required": args.load_required}
    response = requests.post(f"http://{args.endpoint}", json=data)
    output_text = response.text
    print(output_text)
