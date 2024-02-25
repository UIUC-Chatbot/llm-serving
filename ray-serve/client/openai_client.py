# type: ignore
import argparse
import numpy as np
from openai import OpenAI

parser = argparse.ArgumentParser(description="LLM Serving OpenAI Client")
parser.add_argument("-e", "--endpoint", default="http://127.0.0.1:8000/llm")
parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("-s", "--stream", action="store_true")
parser.add_argument("-l", "--loop", action="store_true")
args = parser.parse_args()

openai_api_key = "EMPTY"
client = OpenAI(api_key=openai_api_key, base_url=f"{args.endpoint}/v1/")

prompts = [
    {"role": "user", "content": "Tell me a joke, please."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "user", "content": "What is the meaning of life?"},
    {"role": "user", "content": "What is the best programming language?"},
    {"role": "user", "content": "Tell me a really really long story, please."},
    {
        "role": "user",
        "content": "To be or not to be, that is the question. What is your answer?",
    },
]

while True:
    prompt_index = np.random.randint(0, len(prompts))
    prompt = prompts[prompt_index]
    print(f"\n------------***------------\nPrompt: {prompt['content']}\n")
    res = client.chat.completions.create(
        model=args.model_name,
        messages=[prompt],
        max_tokens=1000,
        stream=args.stream,
    )
    print(f"Response from model {args.model_name}:\n")
    if args.stream:
        for chunk in res:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("")
    else:
        print(res.choices[0].message)

    if not args.loop:
        break
