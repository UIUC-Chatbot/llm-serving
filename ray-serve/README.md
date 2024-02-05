# LLM Serving

## Usage

1. Navigate to the root directory of the project (**ray-serve**).
1. Start Ray cluster in the root directory, e.g. `ray start --head --num-gpus=$(GPU_COUNT)`.  
    `import_path` depends on the location where Ray is started.
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`
1.
    - Users can send requests using OpenAI api.
    - Administrators can directly call the `ModelController` to manage models.

## Architecture

Please read **model_pool.py**.
