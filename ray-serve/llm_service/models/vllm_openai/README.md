# VLLM OpenAI-Compatible Server

As of 2024-2-1, VLLM OpenAI-Compatible Server is implemented as a Python script (with `if __name__ == "__main__"`).

Since Ray only supports deployment of Python classes and functions, not Python scripts, we have to wrap the script in a class. The files in this directory were copied from VLLM and modified to be compatible with Ray Serve.

Some (hopefully) helpful notes:

- The original script uses `argparse` to parse command line arguments. This is not compatible with Ray Serve for whatever reason. Instead, we have to pass the arguments as function arguments, e.g.,  
  `return parser.parse_args(["--served-model-name", model_name, ... ])`
- Some variables are defined in the global scope of the original script, such as the LLM model. We have to move these into the `reconfigure` method of the class. Do not move them into `__init__`, as `__init__` is called when the class is instantiated, and would load the model into memory even if the app is deployed as an inactive replica.
- The model app might be active or inactive. When it is inactive and user request comes in, it has to call the controller to activate itself.
