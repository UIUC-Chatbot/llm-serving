# LLM Serving

## Usage

1. Navigate to the root directory of the project.
1. Start Ray cluster in the root directory, e.g. `ray start --head --num-gpus=$(GPU_COUNT)`.  
    `import_path` depends on the location where Ray is started.
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`
1. Connect to the Model Controller and request models. Model Controller returns the endpoint of the model.
1. Connect to the model endpoint and send requests.
