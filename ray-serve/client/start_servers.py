import asyncio
from aiohttp import ClientSession


async def send_post_request(session: ClientSession, endpoint: str, data: dict):
    print(f"Requesting route for model {data['model_name']}")
    async with session.post(endpoint, json=data) as response:
        response_text = await response.text()
        print(response_text)
        return response_text


endpoint = "http://127.0.0.1:8000/llm/admin"
models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "facebook/opt-13b",
    "gpt2-xl",
    "meta-llama/Llama-2-7b-chat-hf",
    "gpt2-xl",
    "gpt2",
]
requests_data = [
    {
        "key": "IloveRocknRoll",
        "mode": "get",
        "model_name": model,
        "model_type": "vllm_openai",
    }
    for model in models
]

print("Starting servers...")


async def main():
    async with ClientSession() as session:
        tasks = [send_post_request(session, endpoint, data) for data in requests_data]
        await asyncio.gather(*tasks)


asyncio.run(main())
