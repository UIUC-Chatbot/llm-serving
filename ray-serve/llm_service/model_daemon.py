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

    def __init__(
        self,
        controller: str,
        health_check_period: int,
        gpu_check_period: int,
        dump_period: int,
    ) -> None:
        time.sleep(90)  # ModelController might take a while to start, wait for it
        self._controller: DeploymentHandle = serve.get_app_handle(controller)
        self._logger: Logger = getLogger("ray.serve")
        self._watch_list: dict[str, int] = {}

        self._logger.info(f"Daemon initialized with controller {controller}")
        asyncio.create_task(self._check_service_status(health_check_period))
        asyncio.create_task(self._count_gpus(gpu_check_period))
        asyncio.create_task(
            self._dump_current_config("current_config.yaml", dump_period)
        )

    async def _count_gpus(self, check_period: int) -> None:
        while True:
            num_total_gpus: int = await self._controller.get_num_gpus.remote()
            gpus_available_in_ray: int = ray.cluster_resources().get("GPU", 0)
            if num_total_gpus != gpus_available_in_ray:
                self._logger.info(
                    f"LLM service has claimed {num_total_gpus} GPUs. There are {gpus_available_in_ray} GPUs available in Ray, updating service."
                )
                await self._controller.set_num_gpus.remote(gpus_available_in_ray)
            await asyncio.sleep(check_period)

    async def _check_service_status(self, check_period: int):
        while True:
            app_status: dict = serve.status().applications
            for app_name, app in app_status.items():
                is_good: bool = True
                if app.status == "UNHEALTHY":
                    is_good = False
                    self._logger.info(f"App {app_name} is unhealthy.")
                elif app.status == "DEPLOY_FAILED":
                    is_good = False
                    self._logger.info(f"App {app_name} has failed to deploy.")

                if is_good:
                    self._watch_list.pop(app_name, None)
                else:
                    self._watch_list[app_name] = self._watch_list.get(app_name, 0) + 1
                    if self._watch_list[app_name] >= 3:
                        self._logger.warning(
                            f"App {app_name} is still unhealthy after 3 checks, remove it."
                        )
                        await self._controller.delete_model_by_app_name.remote(app_name)
                        self._watch_list.pop(app_name)
                        self._logger.warning(
                            f"Unhealthy App {app_name} has been removed."
                        )

            await asyncio.sleep(check_period)

    async def _dump_current_config(self, dump_path: str, dump_period: int) -> None:
        while True:
            current_config: dict = await self._controller.get_current_config.remote()
            with open(dump_path, "w") as f:
                yaml.dump(current_config, f, sort_keys=False)
            await asyncio.sleep(dump_period)


class _DaemonArgs(BaseModel):
    controller: str
    health_check_period: int  # time period between health checks, in seconds
    gpu_check_period: int  # time period between GPU checks, in seconds
    dump_period: int  # time period between config file dumps, in seconds


def app_builder(args: _DaemonArgs) -> Application:
    return Daemon.bind(
        args.controller,
        args.health_check_period,
        args.gpu_check_period,
        args.dump_period,
    )
