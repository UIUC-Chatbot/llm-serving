import fastapi
from model_app import ModelAppInterface, ModelAppArgs
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import time
from typing import Any

app = fastapi.FastAPI()


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):
    def __init__(self, model_name: str, controller: str, gpus_per_replica: int) -> None:
        self._controller_app: DeploymentHandle = serve.get_app_handle(controller)
        self._is_active: bool = False
        self._model_name: str = model_name
        self._gpus_per_replica: int = gpus_per_replica
        self._last_served_time = time.time()

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            return True
        await self._controller_app.handle_unavailable_model.remote(self._model_name)
        return False

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._is_active: bool = config["is_active"]
        if self._is_active:
            print("Embedding model is active")
        else:
            print("Embedding model is not active")

    def collect_eviction_defense_metrics(self) -> dict[str, Any]:
        return {"last_served_time": self._last_served_time}


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.controller, args.gpus_per_replica)
