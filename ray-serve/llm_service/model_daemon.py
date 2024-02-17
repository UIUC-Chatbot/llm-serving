import asyncio
from fastapi import FastAPI
from logging import getLogger, Logger
from pydantic import BaseModel
import ray
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import time
import yaml

from model_context import ModelContext, ModelStatus

daemon_app = FastAPI()


@serve.deployment(
    name="Daemon",
    ray_actor_options={"num_cpus": 1, "resources": {"head_agents": 1}},
)
@serve.ingress(daemon_app)
class Daemon:
    """
    This class should be deployed on the head node, since it periodically communicates with the
    ModelController.
    """

    def __init__(self, controller: str, check_period: int, dump_period: int) -> None:
        time.sleep(60)  # ModelController might take a while to start, wait for it
        self._check_period: int = check_period
        self._dump_period: int = dump_period
        self._controller: DeploymentHandle = serve.get_app_handle(controller)
        self._logger: Logger = getLogger("ray.serve")
        self._num_health_checks: int = 0
        self._num_config_dumps: int = 0

        self._logger.info(f"Daemon initialized with controller {controller}")
        asyncio.create_task(self._check_service_health(self._check_period))
        asyncio.create_task(
            self._dump_current_config("current_config.yaml", self._dump_period)
        )

    async def _check_service_health(self, check_period: int):
        while True:
            if self._num_health_checks % 20 == 0:
                self._logger.info("Daemon is checking model health.")

            service_info: dict = await self._controller.get_service_info.remote()
            cur_gpus: int = service_info["num_gpus"]
            model_pool: dict[str, ModelContext] = service_info["model_pool"]

            gpus_available_in_ray: int = ray.cluster_resources().get("GPU", 0)
            if gpus_available_in_ray != cur_gpus:
                if gpus_available_in_ray < cur_gpus:
                    self._logger.warning(
                        f"GPUs requested by the service ({cur_gpus}) exceed available GPUs in Ray ({gpus_available_in_ray})."
                    )
                else:
                    self._logger.info(
                        f"Found {gpus_available_in_ray} GPUs in Ray, updating service to use them."
                    )
                await self._controller.set_num_gpus.remote(gpus_available_in_ray)

            unhealthy_models: list[ModelContext] = []
            for model in model_pool.values():
                model_status: ModelStatus = await model.check_model_status()
                if (
                    model_status == ModelStatus.UNHEALTHY
                    or model_status == ModelStatus.DEPLOY_FAILED
                ):
                    unhealthy_models.append(model)
            if unhealthy_models:
                self._logger.warning(
                    f"Unhealthy models: {[model.model_name for model in unhealthy_models]}"
                )
            self._num_health_checks += 1
            await asyncio.sleep(check_period)

    async def _dump_current_config(self, dump_path: str, dump_period: int) -> None:
        while True:
            if self._num_health_checks % 20 == 0:
                self._logger.info("Daemon is dumping current config.")
            current_config: dict = await self._controller.get_current_config.remote()
            with open(dump_path, "w") as f:
                yaml.dump(current_config, f, sort_keys=False)
            self._num_config_dumps += 1
            await asyncio.sleep(dump_period)


class _DaemonArgs(BaseModel):
    controller: str
    check_period: int  # time period between health checks, in seconds
    dump_period: int  # time period between config file dumps, in seconds


def app_builder(args: _DaemonArgs) -> Application:
    return Daemon.bind(args.controller, args.check_period, args.dump_period)
