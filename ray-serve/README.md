# LLM Serving

## Usage

1. Navigate to **/ray-serve/**.
1. Start Ray cluster, e.g. `ray start --head --disable-usage-stats --resources='{"head_agents": 2}' --num-gpus=$(GPU_COUNT)`.  
    - `head_agent` is a custom resource that is used to force some important processes to be deployed on the head node.
    - `import_path` depends on the location where Ray was started.
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`
1.
    - Users can send requests using OpenAI api.
    - Administrators can directly manage the model pool by using **admin_client**.

## Model Reference File

Some models are extremely large and cannot fit into a single GPU. We maintain a reference file which contains the model name and the number of GPUs required to serve the model.

## Architecture

Please read **/llm_service/model_pool.py**.
