import argparse
from client.admin_client import AdminClient
from openai import OpenAI
import time

parser = argparse.ArgumentParser(description="Measure time to first token from a model")
parser.add_argument("-e", "--endpoint", default="http://localhost:5004/llm")
parser.add_argument(
    "--model-name",
    help="Model Name",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
)
args = parser.parse_args()


openai_api_key = "EMPTY"
client = OpenAI(api_key=openai_api_key, base_url=f"{args.endpoint}/v1/")

res = client.chat.completions.create(
    model=args.model_name,
    messages=[{"role": "user", "content": "Tell me a joke, please."}],
    max_tokens=100,
)

start_time = time.time()
res = client.chat.completions.create(
    model=args.model_name,
    messages=[{"role": "user", "content": "Tell me 10 jokes, please."}],
    max_tokens=1000,
    stream=True,
)

for idx, chunk in enumerate(res):
    if idx == 0:
        end_time = time.time()
        print(
            f"Time to first token from {args.model_name}: {end_time - start_time} seconds.\n"
        )
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
