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
        num_gpus: int,
        hf_key: str | None,
        force: bool | None,
    ) -> str:
        data = {
            "key": self._key,
            "model_name": model_name,
            "model_type": model_type,
            "gpus_per_replica": num_gpus,
        }
        if hf_key is not None:
            data["hf_key"] = hf_key
        if force is not None:
            data["force_load"] = force
        response = requests.post(f"{self._endpoint}/get_model", json=data)
        return response.text

    def delete_model(self, model_name: str) -> str:
        data = {"key": self._key, "model_name": model_name}
        response = requests.post(f"{self._endpoint}/delete_model", json=data)
        return response.text

    def show_info(self) -> dict:
        data = {"key": self._key}
        response = requests.post(f"{self._endpoint}/info", json=data)
        return response.json()

    def dump_config(self, config_dump_path: str) -> None:
        data = {"key": self._key, "mode": "dump_config"}
        response = requests.post(f"{self._endpoint}/dump_config", json=data)
        config_file = response.json()
        with open(config_dump_path, "w") as f:
            yaml.dump(config_file, f, sort_keys=False)

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
        help="0: get model route; 1: delete model; 2: reset LLM service; 3: show service info; 4: dump config",
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
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--hf-key", type=str, default=None)
    parser.add_argument("-f", "--force", help="Force Load", default=None)
    parser.add_argument("--config-dump-path", type=str, default="latest_config.yaml")
    args = parser.parse_args()

    admin_client = AdminClient(args.endpoint, args.key)

    if args.mode == 0:
        print(f"Requesting route for model {args.model_name}")
        res = admin_client.get_model_route(
            args.model_name, args.model_type, args.num_gpus, args.hf_key, args.force
        )
        print(res)

    elif args.mode == 1:
        print(f"Deleting model {args.model_name}")
        res = admin_client.delete_model(args.model_name)
        print(res)

    elif args.mode == 2:
        print("Resetting LLM service")
        res = admin_client.reset_llm_service()
        print(res)

    elif args.mode == 3:
        print("Showing LLM service info")
        res = admin_client.show_info()
        print(res)

    elif args.mode == 4:
        print(f"Dumping config to file {args.config_dump_path}")
        admin_client.dump_config(args.config_dump_path)

    else:
        print("Invalid mode")
