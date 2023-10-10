# llm-server
Serve LLMs on NCSA hardware. Support the best FOSS models, and the long tail on HuggingFace Hub.

## Usage

:star:️ https://api.kastan.ai/ :star:

_Free & unbelievably easy LLaMA-2 inference for everyone at NCSA!_

* It’s an API: I host it, you use it. Quick and easy for jobs big and small.
* Access it however you like: Python client, Curl/Postman, or a full web interface playground.
* It’s running on NCSA Center of AI Innovation GPUs, and is fully private & secure thanks to https connections via Zero Trust CloudFlare Tunnels.
* It works with LangChain :parrot::link:

Beautiful implementation detail: it’s a perfect clone of the OpenAI API, making my version a drop in replacement for any openAI calls. Say goodbye to huge OpenAI bills!:moneybag:

:scroll: I wrote [beautiful usage docs & examples here](https://kastanday.notion.site/LLM-Serving-on-prem-OpenAI-Clone-bb06028266d842b0872465f552684177) :eyes: It literally couldn’t be simpler to use :innocent:

:snake: In Python, it’s literally this easy:

```python
import openai # pip install openai
openai.api_key = "irrelevant" # must be non-empty

# 👉 ONLY CODE CHANGE: use our GPUs instead of OpenAI's 👈
openai.api_base = "https://api.kastan.ai/v1"

# exact same api as normal!
completion = openai.Completion.create(
    model="llama-2-7b",
    prompt="What's the capitol of France?",
    max_tokens=200,
    temperature=0.7,
    stream=True)

# ⚡️⚡️ streaming
for token in completion:
  print(token.choices[0].text, end='')
```

:globe_with_meridians: Or from the command line:
```bash
curl https://api.kastan.ai/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{ "prompt": "What is the capital of France?", "echo": true }'
```
## UX Design Goals
1. Flawless API support for the best LLM of the day. An exact clone of the OpenAI API, making it a drop-in replacement.
2. Support for 100% of the models on HuggingFace Hub, some will be easier to use than others.

### Towards 100% Coverage of HuggingFace Hub
⭐️ S-Tier: For the best text LLM of the day, currently LLaMA-2 or Mistral, we offer persistant, ultra-low-latency inference with customized, fused, cuda kernels. This is suitable to build other applications on top of. Any app can now easily and reliably benefit from intelligence.

🥇 A-Tier: If you want a particular LLM, in the list of popular supported ones, that's fine too. They all have optimized inference cuda kernels.

<img width="1713" alt="llm_sever_priorities" src="https://github.com/KastanDay/llm-server/assets/13607221/14c440db-8cff-4b00-9338-47bf839e768e">


## Technical Design


<img width="1666" alt="api kastan ai_routing_design" src="https://github.com/KastanDay/llm-server/assets/13607221/038c129f-dcad-4d84-9b4d-a7fd00148555">


