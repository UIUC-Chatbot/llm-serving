import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description="LLM-Serving OpenAI Client Program")
parser.add_argument("--endpoint", help="LLM-Serving Endpoint", type=str, required=True)
parser.add_argument(
    "--model-name", help="Model Name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
)
args = parser.parse_args()

openai_api_key = "EMPTY"
client = OpenAI(
    api_key=openai_api_key,
    base_url=f"http://{args.endpoint}",  # http://127.0.0.1:8001/v1/
)
completion = client.completions.create(
    model=args.model_name, prompt="Can you tell me a joke?"
)
print("Completion result:", completion)
