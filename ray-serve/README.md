# LLM Serving

## Usage

1. Start Ray cluster, e.g. `ray start --head --num-gpus=$(GPU_COUNT)`
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`
