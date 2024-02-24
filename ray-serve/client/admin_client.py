import argparse
from pprint import pprint
import requests
import yaml


class AdminClient:
    def __init__(self, endpoint: str, key: str):
        self._endpoint: str = endpoint
        self._key: str = key

    def get_model_route(
        self, model_name: str, model_type: str, num_gpus: int, force: bool
    ) -> str:
        data = {
            "key": self._key,
            "mode": "get",
            "model_name": model_name,
            "model_type": model_type,
            "gpus_per_replica": num_gpus,
            "force": force,
        }
        response = requests.post(self._endpoint, json=data)
        return response.text

    def delete_model(self, model_name: str) -> str:
        data = {"key": self._key, "mode": "delete", "model_name": model_name}
        response = requests.post(self._endpoint, json=data)
        return response.text

    def list_models(self, print_models: bool = True) -> dict:
        data = {"key": self._key, "mode": "list"}
        response = requests.post(self._endpoint, json=data)
        model_info = response.json()
        if print_models:
            print(f"\nModel Pool: {len(model_info['model_pool'])}")
            pprint(model_info["model_pool"])
            print(f"\nUnsupported Models: {len(model_info['model_unsupported'])}")
            pprint(model_info["model_unsupported"])
        return model_info

    def dump_config(self, config_dump_path: str) -> None:
        data = {"key": self._key, "mode": "dump_config"}
        response = requests.post(self._endpoint, json=data)
        config_file = response.json()
        with open(config_dump_path, "w") as f:
            yaml.dump(config_file, f, sort_keys=False)

    def show_info(self) -> dict:
        data = {"key": self._key, "mode": "info"}
        response = requests.post(self._endpoint, json=data)
        return response.json()

    def reset_unsupported(self) -> str:
        data = {"key": self._key, "mode": "reset_unsupported"}
        response = requests.post(self._endpoint, json=data)
        return response.text

    def reset_llm_service(self) -> str:
        data = {"key": self._key, "mode": "reset_all"}
        response = requests.post(self._endpoint, json=data)
        return response.text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Serving Admin Client")
    parser.add_argument(
        "-k", "--key", help="Admin Key", type=str, default="IloveRocknRoll"
    )
    parser.add_argument(
        "--mode",
        help="0: get model route; 1: delete model; 2: list models; 3: dump config; 4: show service info; 5: reset unsupported models; 6: reset LLM service",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        help="LLM-Serving Endpoint",
        type=str,
        default="https://api.ncsa.ai/llm/admin",
    )
    parser.add_argument(
        "--model-name",
        help="Model Name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--model-type", help="Model Type", type=str, default="vllm_raw")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("-f", "--force", help="Force Load", action="store_true")
    parser.add_argument("--config-dump-path", type=str, default="latest_config.yaml")
    args = parser.parse_args()

    admin_client = AdminClient(args.endpoint, args.key)

    if args.mode == 0:
        print(f"Requesting route for model {args.model_name}")
        res = admin_client.get_model_route(
            args.model_name, args.model_type, args.num_gpus, args.force
        )
        print(res)

    elif args.mode == 1:
        print(f"Deleting model {args.model_name}")
        res = admin_client.delete_model(args.model_name)
        print(res)

    elif args.mode == 2:
        print("Listing all models")
        res = admin_client.list_models(print_models=True)

    elif args.mode == 3:
        print(f"Dumping config to file {args.config_dump_path}")
        admin_client.dump_config(args.config_dump_path)

    elif args.mode == 4:
        print("Showing LLM service info")
        res = admin_client.show_info()
        print(res)

    elif args.mode == 5:
        print("Resetting unsupported models")
        res = admin_client.reset_unsupported()
        print(res)

    elif args.mode == 6:
        print("Resetting LLM service")
        res = admin_client.reset_llm_service()
        print(res)

    else:
        print("Invalid mode")
