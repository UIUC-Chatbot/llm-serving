# LLM Serving

## Usage

1. Install Ray and Ray Serve. `pip install -r config/requirements.txt`.
1. Navigate to **/ray_serving/**.
1. Start Ray cluster, e.g. `ray start --head --disable-usage-stats --resources='{"head_agents": 2}' --num-gpus=$(GPU_COUNT) --temp-dir=$(TMP_DIR) --dashboard-port=$(PORT)`.  
    - `head_agent` is a custom resource that is used to force some important processes to be deployed on the head node.
    - Python `import_path` depends on the location where Ray was started, so make sure to start Ray in the correct directory.
    - You should run this command on the head node or a node that users can access, since Ray deploys network services on this node.
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`. You can use default config file: `serve deploy config/llm_config_default.yaml`.
1. If your server environment is incompatible with Ray autoscaler, e.g., not using cloud service, then you might have to manually start worker nodes and connect them to the head node. For example, login to a worker node and run `ray start --address=$(head_node_address)`. Please refer to [**Ray Cluster Management CLI**](https://docs.ray.io/en/latest/cluster/cli.html) for more information.
1. Open Ray dashboard to view the status of the LLM serving system. You can access the dashboard by visiting `http://127.0.0.1:8265/` (default to 8265) on the head node. (Dashboard might contain private information and shouldn't be exposed to the public)
    - You can view the document for available endpoints at `$(BASE_URL)/llm/docs`, e.g., <https://api.ncsa.ai/llm/docs>
    - Users can send requests using OpenAI api.
    - Administrators can directly manage the model pool by using **admin_client**.

## Admin API
For loading and unloading models. 
See the Swaggar docs here: https://api.ncsa.ai/llm/docs

Higher priority models will evict lower priority models, in case where autoscaler is disabled. Otherwise, we will always try to scale the Ray cluster (to gain more GPU memory) and launch the new model there. However, if the autoscaler failes (maybe unavailable resources), then higher priority models will evict lower priority models.

The default keep-hot time is 10 minutes, after which the model instance will be shut down due to being idle.

* Priority `0` is lowest and default, positive numbers are higher.
* Priority `>=2` the model will always stay hot, never removed.
* All models, regardless of priority, will always be removed in the evening (CST), to save money. Defined in `clean_unpopular_models()` function.

Note: models that require multiple GPUs must be specified in the `model_reference.json` file, otherwise 1 GPU is used by default. Eventually we will make this dynamic.

## Model Reference File

Some models are extremely large and cannot fit into a single GPU. We maintain a reference file which contains the model name and the number of GPUs required to serve the model.

## Architecture

Please read **model_controller.py**.
