import argparse
import requests

parser = argparse.ArgumentParser(description="LLM-Serving Raw Client Program")
parser.add_argument(
    "-e", "--endpoint", help="LLM-Serving Endpoint", type=str, required=True
)
parser.add_argument("--auto", type=bool, default=False)
args = parser.parse_args()

print(f"Sending prompt to endpoint {args.endpoint}")
while True:
    if args.auto:
        input_data = "Can you tell me a joke?"
    else:
        input_data = input("Type something here ...\n")
    data = {"prompt": input_data}
    response = requests.post(args.endpoint, json=data)
    output_text = response.text
    print(output_text)
