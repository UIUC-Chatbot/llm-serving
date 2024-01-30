# VLLM OpenAI-Compatible Server

As of 2024-1-29, VLLM OpenAI-Compatible Server is implemented as a Python script (with `if __name__ == "__main__"`).

Since Ray only supports deployment of Python classes and functions, not Python scripts, we have to wrap the script in a class. The files in this directory were copied from VLLM and modified to be compatible with Ray Serve.
