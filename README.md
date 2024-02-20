# api.NCSA.ai - LLMs for all
:star:️ https://api.NCSA.ai/ :star:

_Free & unbelievably easy LLaMA-2 inference for everyone at NCSA!_

* It’s an API: I host it, you use it. Quick and easy for jobs big and small.
* Access it however you like: Python client, Curl/Postman, or a full web interface playground.
* It’s running on NCSA Center of AI Innovation GPUs, and is fully private & secure thanks to https connections via Zero Trust CloudFlare Tunnels.
* It works with LangChain :parrot::link:

Beautiful implementation detail: it’s a perfect clone of the OpenAI API, making my version a **drop-in replacement for OpenAI calls** (except embeddings). Say goodbye to huge OpenAI bills!:moneybag:

## Usage

:scroll: I wrote [beautiful usage docs & examples here](https://docs.ncsa.ai) :eyes: It literally couldn’t be simpler to use :innocent:

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

<div style="border: 2px solid #f1f1f1; padding: 10px; margin: 10px; border-radius: 10px; background-color: #f9f9f9;">

## UX Design Goals 🎯

1. <div style="background-color: #fff9db; border: 1px solid #ffe600; padding: 10px; margin: 10px 0; border-radius: 5px;">
    🧠⚡️ <strong>Flawless API support for the best LLM of the day.</strong>
    
    An exact clone of the OpenAI API, making it a drop-in replacement.
   </div>

2. <div style="background-color: #dbfff5; border: 1px solid #00ffe6; padding: 10px; margin: 10px 0; border-radius: 5px;">
    🤗 <strong>Support for 100% of the models on HuggingFace Hub.</strong>
    
    Some will be easier to use than others.
   </div>

</div>


### Towards 100% Coverage of HuggingFace Hub
⭐️ S-Tier: For the best text LLM of the day, currently LLaMA-2 or Mistral, we offer persistant, ultra-low-latency inference with customized, fused, cuda kernels. This is suitable to build other applications on top of. Any app can now easily and reliably benefit from intelligence.

🥇 A-Tier: If you want a particular LLM, in [the list of popular supported ones](https://vllm.readthedocs.io/en/latest/models/supported_models.html), that's fine too. They all have optimized inference cuda kernels.

👍 B-Tier: Most models on the HuggingFace Hub, all those that support [`AutoModel()`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html) and/or [`pipeline()`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline). The only downside here is cold starts, download the model & loading it onto a GPU. 

✨ C-Tier: Models that require custom pre/post-processing code, just supply your own `load()` and `run()` functions, typically copy-pasted from the Readme of a HuggingFace model card. Docs to come.

❌ F-Tier: The current status quo: every researcher doing this independently. It's slow, painful and usually extremely compute-wasteful.

<img width="1713" alt="llm_sever_priorities" src="https://github.com/KastanDay/llm-server/assets/13607221/14c440db-8cff-4b00-9338-47bf839e768e">

## Technical Design

Limitations with WIP solutions: 
* Cuda-OOM errors: If your model doesn't fit on our `4xA40 (48 GB)` server we return an error. Coming soon, we should fallback to accelerate ZeRO stage-3 (CPU/Disk offload). And/or allow a flag for quantization, `load_in_8bit=True` or `load_in_4bit=True`.
* Multi-node support. Currently it's only designed to loadbalance within a single node, soon we should use Ray Serve to support arbitrary hetrogeneous nodes.
* Advanced batching -- when the queue contains separate requests for the same model, batch them and run all jobs requesting that model before moving onto the next model (with a max of 15-20 minutes with any one model in memory, if we have other jobs waiting in the queue. This should balance efficiency, i.e. batching, with fairness, i.e. FIFO queuing).

<img width="1666" alt="api kastan ai_routing_design" src="https://github.com/KastanDay/llm-server/assets/13607221/038c129f-dcad-4d84-9b4d-a7fd00148555">


