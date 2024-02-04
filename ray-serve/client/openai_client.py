from openai import OpenAI


openai_api_key = "EMPTY"
client = OpenAI(api_key=openai_api_key, base_url="http://0.0.0.0:5004/llm/v1")

res = client.chat.completions.create(
    model="tiiuae/falcon-40b",
    messages=[{"role": "system", "content": "You are a chatbot."}],
)
print(res)
