import argparse
import asyncio
from aiohttp import ClientSession
import numpy as np

parser = argparse.ArgumentParser(description="Getting Models Test")
parser.add_argument("--endpoint", default="https://api.ncsa.ai/llm/admin")
args = parser.parse_args()


async def send_post_request(session: ClientSession, endpoint: str, data: dict):
    print(f"Requesting route for model {data['model_name']}")
    async with session.post(endpoint, json=data) as response:
        response_text = await response.text()
        print(response_text)
        return response_text


endpoint = f"http://{args.endpoint}"

models = [
    "baichuan-inc/Baichuan-7B",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-12b",
    "facebook/opt-13b",
    "gpt2-xl",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen-7B-Chat",
    "THUDM/chatglm2-6b",
    "THUDM/chatglm3-6b",
]


requests_data = []
for i in range(100):
    model_index = np.random.randint(0, len(models))
    requests_data.append(
        {
            "key": "IloveRocknRoll",
            "mode": "get",
            "model_name": models[model_index],
            "model_type": "vllm_openai",
        }
    )


print("Starting servers...")


async def main():
    async with ClientSession() as session:
        tasks = [send_post_request(session, endpoint, data) for data in requests_data]
        await asyncio.gather(*tasks)


asyncio.run(main())
