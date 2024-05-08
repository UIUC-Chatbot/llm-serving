# VLLM OpenAI-Compatible Server

As of 2024-5-5, vLLM OpenAI-Compatible Server is implemented as a Python script (with `if __name__ == "__main__"`). [openai/api_server.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py) is the file to look into.

Since Ray only supports deployment of Python classes and functions, not Python scripts, we have to wrap the script in a class. The files in this directory were copied from VLLM and modified to be compatible with Ray Serve.

Some (hopefully) helpful notes:

- The original script uses `argparse` to parse command line arguments. This is not compatible with Ray Serve for whatever reason. Instead, we have to pass the arguments as function arguments, e.g.,  
  `return parser.parse_args(["--model", model_name, ... ])`
- Some variables are defined in the global scope of the original script, such as the LLM model. We have to move these into the `reconfigure` method of the class. Do not move them into `__init__`, as `__init__` is called when the class is instantiated, and would load the model into memory even if the app is deployed as an inactive replica.
- The model app might be active or inactive. When it is inactive and user request comes in, it has to call the controller to activate itself.
- If we are using distributed inference, then we need to create a Ray placement group. Pay attention to the cluster initialization logic in VLLM and make sure it is compatible with our system. For example, as of vllm==0.3.3, the ModelApp deployment itself should only acquire one CPU and no GPUs, because VLLM initialization function will deploy Ray workers which acquire all GPUs allocated in the placement group. However, during arguments initialization, if we pass `device` as `auto`, this will cause the driver code to call `torch.cuda.is_available()` to find out if we have a GPU available. But since we allocated no GPUs for the driver code, this will error. Thus, we must pass `device` as `cuda` to avoid this error.
