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

output = ""

for idx, chunk in enumerate(res):
    if idx == 0:
        first_token_time = time.time()
        print(
            f"Time to first token from {args.model_name}: {first_token_time - start_time} seconds."
        )
    if chunk.choices[0].delta.content is not None:
        text = chunk.choices[0].delta.content
        output += text
end_time = time.time()
tokens = output.split()
total_tokens = len(tokens)
time_per_token = (end_time - first_token_time) / (total_tokens - 1)
print(f"Time per output token from {args.model_name}: {time_per_token} seconds.")
print(f"Total tokens: {total_tokens}")
print("\nOutput: ")
print(output)
