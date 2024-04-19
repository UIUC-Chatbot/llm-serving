import argparse
import requests
import yaml


class AdminClient:
    def __init__(self, endpoint: str, key: str):
        self._endpoint: str = endpoint
        self._key: str = key

    def get_model_route(
        self,
        model_name: str,
        model_type: str,
        num_replicas: int,
        gpus_per_replica: int,
        priority: int,
        hf_key: str | None = None,
    ) -> str:
        data = {
            "key": self._key,
            "model_name": model_name,
            "model_type": model_type,
            "num_replicas": num_replicas,
            "gpus_per_replica": gpus_per_replica,
            "priority": priority,
        }
        if hf_key is not None:
            data["hf_key"] = hf_key
        response = requests.post(f"{self._endpoint}/get_model", json=data)
        return response.text

    def delete_model(self, model_name: str) -> str:
        data = {"key": self._key, "model_name": model_name}
        response = requests.post(f"{self._endpoint}/delete_model", json=data)
        return response.text

    def load_model(self, model_name: str, num_replicas: int) -> str:
        data = {
            "key": self._key,
            "model_name": model_name,
            "num_replicas": num_replicas,
        }
        response = requests.post(f"{self._endpoint}/load_model", json=data)
        return response.text

    def show_info(self) -> dict:
        data = {"key": self._key}
        response = requests.post(f"{self._endpoint}/info", json=data)
        return response.json()

    def dump_config(self, config_dump_path: str) -> None:
        data = {"key": self._key}
        response = requests.post(f"{self._endpoint}/dump_config", json=data)
        config_file = response.json()
        with open(config_dump_path, "w") as f:
            yaml.dump(config_file, f, sort_keys=False)

    def load_reference(self, reference_path: str) -> None:
        data = {"key": self._key, "model_reference_path": reference_path}
        response = requests.post(f"{self._endpoint}/load_model_reference", json=data)
        return response.json()

    def reset_llm_service(self) -> str:
        data = {"key": self._key}
        response = requests.post(f"{self._endpoint}/reset", json=data)
        return response.text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Serving Admin Client")
    parser.add_argument(
        "-k", "--key", help="Admin Key", type=str, default="IloveRocknRoll"
    )
    parser.add_argument(
        "--mode",
        help="0: get model route; 1: delete model; 2: load model replicas; 3: reset LLM service; 4: show service info; 5: dump config; 6: load model reference",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        help="LLM-Serving Endpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-name",
        help="Model Name",
        type=str,
        default="unknown",
    )
    parser.add_argument("--model-type", help="Model Type", type=str, default="unknown")
    parser.add_argument("--num_replicas", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--hf-key", type=str, default=None)
    parser.add_argument("--config-dump-path", type=str, default="latest_config.yaml")
    parser.add_argument(
        "--model-reference-path", type=str, default="config/model_reference.json"
    )
    args = parser.parse_args()

    admin_client = AdminClient(args.endpoint, args.key)

    if args.mode == 0:
        print(f"Requesting route for model {args.model_name}")
        res = admin_client.get_model_route(
            args.model_name,
            args.model_type,
            args.num_replicas,
            args.num_gpus,
            args.hf_key,
        )
        print(res)

    elif args.mode == 1:
        print(f"Deleting model {args.model_name}")
        res = admin_client.delete_model(args.model_name)
        print(res)

    elif args.mode == 2:
        print(f"Loading {args.num_replicas} replicas of model {args.model_name}")
        res = admin_client.load_model(args.model_name, args.num_replicas)
        print(res)

    elif args.mode == 3:
        print("Resetting LLM service")
        res = admin_client.reset_llm_service()
        print(res)

    elif args.mode == 4:
        print("Showing LLM service info")
        res = admin_client.show_info()
        print(res)

    elif args.mode == 5:
        print(f"Dumping config to file {args.config_dump_path}")
        admin_client.dump_config(args.config_dump_path)

    elif args.mode == 6:
        print(f"Loading model reference from {args.model_reference_path}")
        res = admin_client.load_reference(args.model_reference_path)
        print(res)

    else:
        print("Invalid mode")
