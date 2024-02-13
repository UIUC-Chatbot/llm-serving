from pprint import pprint
import requests
import yaml


class AdminClient:
    def __init__(self, endpoint: str, key: str):
        self._endpoint: str = endpoint
        self._key: str = key

    def get_model_route(self, model_name: str, model_type: str, num_gpus: int) -> str:
        data = {
            "key": self._key,
            "mode": "get",
            "model_name": model_name,
            "model_type": model_type,
            "gpus_per_replica": num_gpus,
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

    def reset_unsupported(self) -> str:
        data = {"key": self._key, "mode": "reset_unsupported"}
        response = requests.post(self._endpoint, json=data)
        return response.text

    def reset_llm_service(self) -> str:
        data = {"key": self._key, "mode": "reset_all"}
        response = requests.post(self._endpoint, json=data)
        return response.text
