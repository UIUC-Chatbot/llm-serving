# LLM Serving

## Usage

1. Navigate to **/ray-serve/**.
1. Start Ray cluster, e.g. `ray start --head --disable-usage-stats --resources='{"head_agents": 2}' --num-gpus=$(GPU_COUNT)`.  
    - `head_agent` is a custom resource that is used to force some important processes to be deployed on the head node.
    - Python `import_path` depends on the location where Ray was started, so make sure to start Ray in the correct directory.
    - You should run this command on the head node or a node that users can access, since Ray deploys network services on this node.
1. Use [**config file**](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) to specify the model and the serving configuration.
1. Start llm-serving: `serve deploy $(config file)`. You can use default config file: `serve deploy config/llm_config_default.yaml`.
1. If your server environment is incompatible with Ray autoscaler, e.g., not using cloud service, then you might have to manually start worker nodes and connect them to the head node. For example, login to a worker node and run `ray start --address=$(head_node_address)`. Please refer to [**Ray Cluster Management CLI**](https://docs.ray.io/en/latest/cluster/cli.html) for more information.
1. Open Ray dashboard to view the status of the LLM serving system. You can access the dashboard by visiting `http://127.0.0.1:8265/` on the head node. (Dashboard might contain private information and shouldn't be exposed to the public)
    - Users can send requests using OpenAI api.
    - Administrators can directly manage the model pool by using **admin_client**.

## Model Reference File

Some models are extremely large and cannot fit into a single GPU. We maintain a reference file which contains the model name and the number of GPUs required to serve the model.

## Architecture

Please read **model_controller.py**.
