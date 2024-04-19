import fastapi
from logging import getLogger, Logger
from model_app import ModelAppInterface, ModelAppArgs
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import time

app = fastapi.FastAPI()


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):
    """
    An empty model that does nothing, for testing purposes.
    """

    def __init__(self, model_name: str, main: str, gpus_per_replica: int) -> None:
        self._main: DeploymentHandle = serve.get_app_handle(main)
        self._logger: Logger = getLogger("ray.serve")
        self._is_active: bool = False
        self._model_name: str = model_name

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            return True
        await self._main.load_model.remote(self._model_name, 1)
        return False

    def reconfigure(self, config: dict) -> None:
        self._is_active: bool = config["is_active"]

    def collect_eviction_defense_metrics(self) -> dict:
        return {"last_served_time": time.time()}

    @app.post("/")
    async def call(self, request: fastapi.Request) -> str:
        if not await self._check_model_availability():
            return "Model is not active."
        else:
            return "Model is active."


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.main, args.gpus_per_replica)
