import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description="LLM Serving OpenAI Client")
parser.add_argument("--endpoint", default="127.0.0.1:8000/llm")
parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()

openai_api_key = "EMPTY"
client = OpenAI(api_key=openai_api_key, base_url=f"http://{args.endpoint}/v1/")

res = client.chat.completions.create(
    model=args.model_name,
    messages=[{"role": "user", "content": "Tell me a joke, please."}],
    max_tokens=1000,
    stream=args.stream,
)

if args.stream:
    for chunk in res:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    print("")
else:
    print(res.choices[0].message)
