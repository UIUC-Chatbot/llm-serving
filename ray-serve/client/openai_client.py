import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description="LLM Serving OpenAI Client")
parser.add_argument("--endpoint", default="llm")
args = parser.parse_args()

openai_api_key = "EMPTY"
client = OpenAI(
    api_key=openai_api_key, base_url=f"http://0.0.0.0:5004/{args.endpoint}/v1/"
)

res = client.chat.completions.create(
    model="Qwen/Qwen-7B-Chat",
    messages=[{"role": "system", "content": "Tell me a joke, please."}],
    max_tokens=1000,
)
print(res)
