import asyncio
from enum import Enum
from fastapi import FastAPI, Request
from fastapi.responses import Response
from logging import getLogger, Logger
import os
from pydantic import BaseModel
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve import Application
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import ServeDeploySchema
import time
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ErrorResponse,
)
import yaml

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    They have their own endpoints. ModelApp is an abstract class, and the actual mode
    implementation is specified by the model_type.

    When a ModelApp is inactive, users might request to load that model. The ModelApp sends a
    request to the ModelController to load the model. The ModelController might select a victim
    model to evict from GPU and load the requested model.
"""


class _ModelType(Enum):
    VLLM_RAW = 1  # Raw VLLM model, created by llm = LLM(model="model_name")
    VLLM_OPENAI = 2  # VLLM OpenAI-Compatible server


class _ModelContext:
    def __init__(
        self,
        app_name: str,
        model_name: str,
        route_prefix: str,
        model_type: _ModelType,
        gpus_per_replica: int,
    ) -> None:
        self.app_name: str = app_name
        self.app_handle: DeploymentHandle
        self.model_name: str = model_name
        self.model_type: _ModelType = model_type
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment
        self.created_time: int = int(time.time())
        self.gpus_per_replica: int = gpus_per_replica
        self.used_gpus: int = 0

    async def maybe_wait_until_ready(self) -> None:
        """
        Wait until the serve app is deployed.
        """
        start_time = asyncio.get_event_loop().time()
        while not hasattr(self, "app_handle"):
            if asyncio.get_event_loop().time() - start_time > 10:  # Timeout
                raise TimeoutError(f"App for model {self.model_name} doesn't exist.")
            await asyncio.sleep(0.5)


class _ConfigWriter:
    """
    This is a helper class that ModelController uses to update the config file.

    Ray uses config file (yaml) to dynamically update serve apps and deployments.
    ModelController maintains a local copy of the config file, which is updated whenever a model
    is going to be added, removed, activated, or deactivated.
    ModelController sends the config file to the Ray dashboard service, which then updates the
    serve apps and deployments.

    Refer to Ray documentation for more details about the format of the config files.
    """

    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path, "r") as file:
            self._config: dict = yaml.safe_load(file)
        self._apps: list[dict] = self._config["applications"]
        self._logger: Logger = getLogger("ray.serve")

    def _apply_config(self) -> None:
        """
        The following code sends a request containing the latest config file to the Ray
        dashboard, which then updates the serve apps and deployments. The call is async; it
        returns immediately without waiting for the dashboard to finish the update.
        """
        ServeDeploySchema.parse_obj(self._config)
        address: str = os.environ.get("RAY_DASHBOARD_ADDRESS", "http://localhost:8265")
        ServeSubmissionClient(address).deploy_applications(self._config)

    def add_app(self, model: _ModelContext, is_active: bool) -> None:
        # import_path is the path to the actual model implementation.
        if model.model_type == _ModelType.VLLM_RAW:
            import_path = "models.vllm_raw:app_builder"
        elif model.model_type == _ModelType.VLLM_OPENAI:
            import_path = "models.vllm_openai.openai_server:app_builder"
        else:
            raise ValueError("Invalid model type.")

        # VLLM supports distributed inference via Ray. However, this seems to conflict with
        # Ray serve if we explicitly specify num_gpus of that deployment. For now, we just set
        # tensor_parallel_size to the required number and trust VLLM and Ray will do their
        # jobs.
        # For models that don't use distributed inference, we must specify num_gpus == 1. Otherwise,
        # Ray serve will not allocate GPU resources to the deployment.
        if model.gpus_per_replica == 1:
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": {"num_cpus": 1, "num_gpus": model.used_gpus},
                    "user_config": {"is_active": is_active},
                },
            ]
        else:
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": {"num_cpus": 1},
                    "user_config": {"is_active": is_active},
                },
            ]

        # Add the new app to the config file
        self._apps.append(
            {
                "name": model.app_name,
                "route_prefix": model.route_prefix,
                "import_path": import_path,
                "args": {
                    "model_name": model.model_name,
                    # ModelController is the first app in the config file.
                    "controller": self._config["applications"][0]["name"],
                    "gpus_per_replica": model.gpus_per_replica,
                },
                "deployments": deployments,
            }
        )

        self._apply_config()
        self._logger.info(f"App: {model.app_name} added.")

    def remove_app(self, model: _ModelContext) -> None:
        self._config["applications"] = [
            app for app in self._apps if app.get("name") != model.app_name
        ]
        self._apps = self._config["applications"]
        self._apply_config()
        self._logger.info(f"App: {model.app_name} removed.")

    def activate_app(self, model: _ModelContext) -> None:
        app = next(
            (d for d in self._apps if d.get("name") == model.app_name),
            None,
        )
        if app is None:
            raise ValueError(f"App {model.app_name} not found.")
        app["deployments"][0]["user_config"]["is_active"] = True
        if model.gpus_per_replica == 1:
            app["deployments"][0]["ray_actor_options"]["num_gpus"] = 1
        else:
            # Distributed inference shouldn't use num_gpus, see comments in _add_app() for details.
            app["deployments"][0]["ray_actor_options"].pop("num_gpus", None)

        self._apply_config()
        self._logger.info(f"App: {model.app_name} activated.")

    def deactivate_app(self, model: _ModelContext) -> None:
        app = next(
            (d for d in self._apps if d.get("name") == model.app_name),
            None,
        )
        if app is None:
            raise ValueError(f"App {model.app_name} not found.")
        app["deployments"][0]["user_config"]["is_active"] = False
        app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0

        self._apply_config()
        self._logger.info(f"App: {model.app_name} deactivated.")

    def dump_config(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self._config, file, sort_keys=False)


class AdminRequest(BaseModel):
    key: str
    mode: str
    model_name: str
    model_type: str = "vllm_openai"
    gpus_per_replica: int = 1
    config_dump_path: str = "latest_config.yaml"


main_app = FastAPI()


@serve.deployment
@serve.ingress(main_app)
class ModelController:
    def __init__(self, config_file_path: str, num_gpus: int) -> None:
        self._config_writer: _ConfigWriter = _ConfigWriter(config_file_path)
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, _ModelContext] = {}
        self._num_gpus: int = num_gpus
        self._num_served_models: int = 0

        # Lock required for modifying data structures
        self._lock: asyncio.Lock = asyncio.Lock()

    def _has_available_gpu(self, required_gpus: int) -> bool:
        """
        Check if there is any available GPUs.
        """
        used_gpus: int = 0
        for model_name in self._model_pool.keys():
            used_gpus += self._model_pool[model_name].used_gpus
        return self._num_gpus >= used_gpus + required_gpus

    async def _get_or_register_model(
        self, model_name: str, model_type: _ModelType, gpus_per_replica: int
    ) -> _ModelContext:
        """
        Return the model_context of the requested model. Create a serve app for the model if it's
        not created yet.
        """
        if model_name in self._model_pool:
            await self._model_pool[model_name].maybe_wait_until_ready()
            return self._model_pool[model_name]

        async with self._lock:
            if model_name in self._model_pool:
                return self._model_pool[model_name]

            model_context = _ModelContext(
                app_name=model_name.replace("/", "---"),
                model_name=model_name,
                route_prefix=f"/model-{self._num_served_models}",
                model_type=model_type,
                gpus_per_replica=gpus_per_replica,
            )
            self._num_served_models += 1
            if self._has_available_gpu(gpus_per_replica):
                model_context.used_gpus = gpus_per_replica
                self._config_writer.add_app(model=model_context, is_active=True)
            else:
                model_context.used_gpus = 0
                self._config_writer.add_app(model=model_context, is_active=False)
            self._model_pool[model_name] = model_context

            # We just created the app and now we get the app handle to it.
            # Note that app updates via config file are async, so we might need to wait for a while
            # before the app is actually created.
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    app_handle = serve.get_app_handle(model_context.app_name)
                    model_context.app_handle = app_handle
                    return self._model_pool[model_name]
                except RayServeException:
                    if asyncio.get_event_loop().time() - start_time > 10:  # Timeout
                        raise TimeoutError(f"App for model {model_name} doesn't exist.")
                    await asyncio.sleep(0.5)

    async def _delete_model(self, model_name: str) -> bool:
        """
        Delete the model app with the given name.
        """
        if model_name in self._model_pool:
            async with self._lock:
                self._config_writer.remove_app(model=self._model_pool[model_name])
                self._model_pool.pop(model_name)
                return True
        else:
            return False

    def _load_or_replace_model(
        self, model_in: _ModelContext, model_out: None | _ModelContext = None
    ) -> None:
        """
        Loads model_in into GPU. Unloads model_out from GPU if necessary.
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify whether deactivating model_out actually releases GPU
        resources, nor does it check for the availability of any GPU.
        """
        if model_out is not None:
            self._config_writer.deactivate_app(model_out)
            model_out.used_gpus = 0
        self._config_writer.activate_app(model_in)
        model_in.used_gpus = model_in.gpus_per_replica

    async def _select_victim(
        self, initiator: _ModelContext, required_gpus: int
    ) -> _ModelContext:
        """
        The initiator model has requested to load itself into GPU. However, there is no available
        GPU. This function selects a victim model to evict from GPU.
        """
        candidates = []
        for model in self._model_pool.values():
            if model.app_name == initiator.app_name:
                continue
            # TODO: we can remove more than one model at a time if they are not used.
            if model.used_gpus < required_gpus:
                continue
            # TODO: what if the app is not ready yet? So far, hasn't run into issues.
            metrics = await model.app_handle.collect_eviction_defense_metrics.remote()
            candidates.append((model, metrics["last_served_time"]))
        # Remove the least recently used model
        victim = min(candidates, key=lambda x: x[1])[0]
        # TODO: do we need to wait here for a while?
        return victim

    async def handle_unavailable_model(self, model_name: str) -> None:
        """
        This function is called by an inactive ModelApp who wants to load itself into GPU.
        """
        model = self._model_pool[model_name]
        async with self._lock:
            if model.used_gpus > 0:
                return  # If the model is already loaded, ignore this request.
            if self._has_available_gpu(model.gpus_per_replica):
                # If there is an available GPU, load the model.
                return self._load_or_replace_model(model)
            # If there is no available GPU, evict a model from GPU and load the requested model.
            victim: _ModelContext = await self._select_victim(
                model, model.gpus_per_replica
            )
            return self._load_or_replace_model(model, victim)

    """
    OpenAI-ish API endpoints
    """

    @main_app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @main_app.get("/v1/models")
    async def show_available_models(self):
        def model_dump(models: list[dict]):
            for model in self._model_pool.values():
                model_info: dict = {
                    "id": model.model_name,
                    "object": "model",
                    "created": model.created_time,
                    "owned_by": "NCSA",
                }
                models.append(model_info)

        models = []
        model_dump(models)
        return {"object": "list", "data": models}

    @main_app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if request.model in self._model_pool:
            self._logger.info(f"Model {request.model} found.")
            return "Model found."
        else:
            return ErrorResponse(
                message=f"The model `{request.model}` does not exist.",
                type="NotFoundError",
                code=404,
            )

    """
    Admin API endpoints
    """

    @main_app.post("/admin")
    async def admin_call(self, request: AdminRequest) -> str:
        # TODO: the key is currently visible on GitHub. We need to change this.
        if request.key != "IloveRocknRoll":
            return "Permission denied. Aborting."
        if request.mode == "get":
            if request.model_type == "vllm_raw":
                model_type = _ModelType.VLLM_RAW
            elif request.model_type == "vllm_openai":
                model_type = _ModelType.VLLM_OPENAI
            else:
                return "Invalid model type. Aborting."
            model_context = await self._get_or_register_model(
                model_name=request.model_name,
                model_type=model_type,
                gpus_per_replica=request.gpus_per_replica,
            )
            return f"Model {model_context.model_name} endpoint: {model_context.route_prefix}"
        elif request.mode == "delete":
            if await self._delete_model(request.model_name):
                return f"Model {request.model_name} deleted."
            else:
                return f"Model {request.model_name} not found."
        elif request.mode == "dump_config":
            self._config_writer.dump_config(request.config_dump_path)
            return f"Config dumped to {request.config_dump_path}"
        else:
            return "Invalid mode. Aborting."


class ControllerArgs(BaseModel):
    config_file_path: str
    num_gpus: int


def app_builder(args: ControllerArgs) -> Application:
    return ModelController.bind(args.config_file_path, args.num_gpus)
