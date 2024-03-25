import asyncio
import datetime
from fastapi import FastAPI
from logging import getLogger, Logger
from pydantic import BaseModel
import ray
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import subprocess
import time
import yaml

from model_context import ModelType


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
        main: str,
        clean_period: int,
        dump_period: int,
        gpu_check_period: int,
        health_check_period: int,
    ) -> None:
        time.sleep(90)  # ModelController might take a while to start, wait for it
        self._main: DeploymentHandle = serve.get_app_handle(main)
        self._logger: Logger = getLogger("ray.serve")
        self._watch_list: dict[str, int] = {}

        self._logger.info(f"Daemon initialized with controller {main}")

        if clean_period > 0:
            self._logger.info(
                f"Cleaning unpopular models every {clean_period} seconds."
            )
            asyncio.create_task(self._clean_unpopular_models(clean_period))

        if dump_period > 0:
            self._logger.info(f"Dumping current config every {dump_period} seconds.")
            asyncio.create_task(
                self._dump_current_config("config/current_config.yaml", dump_period)
            )

        if gpu_check_period > 0:
            self._logger.info(f"Watching GPUs every {gpu_check_period} seconds.")
            asyncio.create_task(self._watch_gpus(gpu_check_period))

        if health_check_period > 0:
            self._logger.info(
                f"Checking service status every {health_check_period} seconds."
            )
            asyncio.create_task(self._check_service_status(health_check_period))

    async def _watch_gpus(self, check_period: int) -> None:
        self._num_gpus: int = await self._main.update_num_gpus.remote()
        scale_up: bool = False
        while True:
            try:
                gpus_available_in_ray: int = ray.cluster_resources().get("GPU", 0)
                if self._num_gpus != gpus_available_in_ray:
                    self._logger.info(
                        f"LLM service has claimed {self._num_gpus} GPUs. There are {gpus_available_in_ray} GPUs available in Ray, updating service."
                    )
                    self._num_gpus = await self._main.update_num_gpus.remote()

                # Hardcode an autoscaler for slurm, dirty implementation, urrrgh
                avail_gpus: int = await self._main.count_available_gpus.remote()
                if avail_gpus < 0:
                    if not scale_up:
                        self._logger.info("Needs more GPUs, scaling up.")
                        subprocess.run(
                            ["sbatch", "config/node.sh"], text=True, capture_output=True
                        )
                        scale_up = True
                else:
                    scale_up = False
                # Hardcode finished
            except Exception as e:
                self._logger.error(f"Error when checking GPUs: {e}")

            await asyncio.sleep(check_period)

    async def _check_service_status(self, check_period: int) -> None:
        while True:
            try:
                self._logger.debug(f"Watch list: {self._watch_list}")
                app_status: dict = serve.status().applications

                # Remove apps from watch list if they no longer exist
                apps_to_ignore = [
                    app for app in self._watch_list if app not in app_status
                ]
                for app in apps_to_ignore:
                    self._watch_list.pop(app)

                # Check if any apps are unhealthy
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
                        self._watch_list[app_name] = (
                            self._watch_list.get(app_name, 0) + 1
                        )
                        if self._watch_list[app_name] >= 3:
                            self._logger.warning(
                                f"App {app_name} is still unhealthy after 3 checks, remove it."
                            )
                            await self._main.delete_model_by_app_name.remote(app_name)
                            self._watch_list.pop(app_name)
                            self._logger.warning(
                                f"Unhealthy App {app_name} has been removed."
                            )
            except Exception as e:
                self._logger.error(f"Error when checking service status: {e}")

            await asyncio.sleep(check_period)

    async def _clean_unpopular_models(self, check_period: int) -> None:
        while True:
            try:
                self._logger.info("Cleaning unpopular models.")
                self._main.clean_unpopular_models.remote()

                now = datetime.datetime.now().time()
                morning = datetime.time(9, 0, 0)
                elapsed = datetime.time(9, 40, 0)
                if now > morning and now < elapsed:
                    self._logger.info("Activate low-latency model.")
                    self._main.get_or_register_model.remote(
                        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                        ModelType.VLLM_OPENAI,
                        2,
                        2,
                    )
            except Exception as e:
                self._logger.error(f"Error when cleaning unpopular models: {e}")

            await asyncio.sleep(check_period)

    async def _dump_current_config(self, dump_path: str, dump_period: int) -> None:
        while True:
            try:
                current_config: dict = await self._main.get_current_config.remote()
                with open(dump_path, "w") as f:
                    yaml.dump(current_config, f, sort_keys=False)
            except Exception as e:
                self._logger.error(f"Error when dumping current config: {e}")

            await asyncio.sleep(dump_period)


class _DaemonArgs(BaseModel):
    main: str
    clean_period: int  # time period between model cleanups, in seconds
    dump_period: int  # time period between config file dumps, in seconds
    gpu_check_period: int  # time period between GPU checks, in seconds
    health_check_period: int  # time period between health checks, in seconds


def app_builder(args: _DaemonArgs) -> Application:
    return Daemon.bind(
        args.main,
        args.clean_period,
        args.dump_period,
        args.gpu_check_period,
        args.health_check_period,
    )
