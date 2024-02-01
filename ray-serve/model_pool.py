import asyncio
from enum import Enum
import fastapi
from logging import getLogger, Logger
import os
from pydantic import BaseModel
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import ServeDeploySchema
import yaml

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    ModelApp is an abstract class, and the actual model implementation is specified by the
    model_type.

    When a ModelApp is inactive, users might request to load that model. The ModelApp sends a
    request to the ModelController to load the model. The ModelController might select a victim
    model to evict from GPU and load the requested model.
"""


class _AppAction(Enum):
    ADD = 1  # Add a new serve app
    ADD_LOAD = 2  # Add a new serve app and load it immediately
    REMOVE = 3  # Remove a serve app
    UPDATE = 4  # Update a serve app


class _DeploymentAction(Enum):
    ACTIVATE = 1  # Activate a deployment
    DEACTIVATE = 2  # Deactivate a deployment


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
        self.app_handle: DeploymentHandle | None = None
        self.model_name: str = model_name
        self.model_type: _ModelType = model_type
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment
        self.gpus_per_replica: int = gpus_per_replica
        self.used_gpus: int = 0


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

    def _add_app(self, model: _ModelContext, is_active: bool) -> None:
        # import_path is the path to the actual model implementation.
        if model.model_type == _ModelType.VLLM_RAW:
            import_path = "vllm_raw:app_builder"
        elif model.model_type == _ModelType.VLLM_OPENAI:
            import_path = "vllm_server.openai.vllm_openai:app_builder"

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

    def _remove_app(self, model: _ModelContext) -> None:
        self._config["applications"] = [
            app for app in self._apps if app.get("name") != model.app_name
        ]
        self._apps = self._config["applications"]

    def _activate_app(self, model: _ModelContext) -> None:
        app = next(
            (d for d in self._apps if d.get("name") == model.app_name),
            None,
        )
        app["deployments"][0]["user_config"]["is_active"] = True
        if model.gpus_per_replica == 1:
            app["deployments"][0]["ray_actor_options"]["num_gpus"] = 1
        else:
            # Distributed inference shouldn't use num_gpus, see comments in _add_app() for details.
            app["deployments"][0]["ray_actor_options"].pop("num_gpus", None)

    def _deactivate_app(self, model: _ModelContext) -> None:
        app = next(
            (d for d in self._apps if d.get("name") == model.app_name),
            None,
        )
        app["deployments"][0]["user_config"]["is_active"] = False
        # We set num_gpus to 0 for distributed inference, too, to force a redeployment.
        app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0

    def dump_config(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self._config, file, sort_keys=False)

    def update_config(
        self,
        model: _ModelContext,
        app_action: _AppAction,
        deploy_action: _DeploymentAction | None = None,
    ) -> None:
        if app_action == _AppAction.ADD:
            self._add_app(model, False)
        elif app_action == _AppAction.ADD_LOAD:
            self._add_app(model, True)
        elif app_action == _AppAction.REMOVE:
            self._remove_app(model)
        elif app_action == _AppAction.UPDATE:
            if deploy_action == _DeploymentAction.ACTIVATE:
                self._activate_app(model)
            elif deploy_action == _DeploymentAction.DEACTIVATE:
                self._deactivate_app(model)

        # The following code sends a request containing the updated config file to the Ray
        # dashboard, which then updates the serve apps and deployments. The call is async; it
        # returns immediately without waiting for the dashboard to finish the update.
        ServeDeploySchema.parse_obj(self._config)
        address: str = os.environ.get("RAY_DASHBOARD_ADDRESS", "http://localhost:8265")
        ServeSubmissionClient(address).deploy_applications(self._config)

        # Log the update
        deploy_status: str = f", Deployment: {deploy_action}" if deploy_action else ""
        self._logger.info(f"App: {model.app_name}, Action: {app_action}{deploy_status}")


class UserRequest(BaseModel):
    mode: str
    model_name: str
    model_type: str | None = None
    gpus_per_replica: int = 1
    file_path: str | None = None


main_app = fastapi.FastAPI()


@serve.deployment
@serve.ingress(main_app)
class ModelController:
    def __init__(self, config_file_path: str, num_gpus: int) -> None:
        self._config_writer: _ConfigWriter = _ConfigWriter(config_file_path)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, _ModelContext] = {}
        self._num_gpus: int = num_gpus
        self._num_served_models: int = 0

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
    ) -> str:
        """
        Return the route prefix of the requested model. Create a serve app for the model if it's
        not created yet.
        """
        async with self._lock:
            if model_name not in self._model_pool:
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
                    self._config_writer.update_config(
                        model=model_context, app_action=_AppAction.ADD_LOAD
                    )
                else:
                    model_context.used_gpus = 0
                    self._config_writer.update_config(
                        model=model_context, app_action=_AppAction.ADD
                    )
                self._model_pool[model_name] = model_context
            route = f"Route prefix: {self._model_pool[model_name].route_prefix}"
            server_type = f"Server type: {model_type}"
            return f"{route}, {server_type}"

    async def _delete_model(self, model_name: str) -> str:
        """
        Delete the model app with the given name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                self._config_writer.update_config(
                    model=self._model_pool[model_name], app_action=_AppAction.REMOVE
                )
                self._model_pool.pop(model_name)
                return f"Model {model_name} deleted."
            else:
                return f"Model {model_name} not found."

    @main_app.post("/")
    async def call(self, request: UserRequest) -> str:
        """
        The entrypoint of the ModelController. The input should be conform to the UserRequest schema.
        """
        mode: str = request.mode
        model_name: str = request.model_name

        if mode == "get":
            model_type: str = request.model_type
            if model_type == "vllm_raw":
                model_type = _ModelType.VLLM_RAW
            elif model_type == "vllm_openai":
                model_type = _ModelType.VLLM_OPENAI
            else:
                return "Invalid model type. Aborting."
            num_gpus: int = request.gpus_per_replica
            return await self._get_or_register_model(model_name, model_type, num_gpus)
        elif mode == "delete":
            return await self._delete_model(model_name)
        elif mode == "dump_config":
            file_path: str = request.file_path
            self._config_writer.dump_config(file_path)
            return f"Config dumped to {file_path}"
        else:
            return "Invalid mode. Aborting."

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
            self._config_writer.update_config(
                model=model_out,
                app_action=_AppAction.UPDATE,
                deploy_action=_DeploymentAction.DEACTIVATE,
            )
            model_out.used_gpus = 0
        self._config_writer.update_config(
            model=model_in,
            app_action=_AppAction.UPDATE,
            deploy_action=_DeploymentAction.ACTIVATE,
        )
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
            if model.app_handle is None:
                model.app_handle = serve.get_app_handle(model.app_name)
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
        async with self._lock:
            model = self._model_pool[model_name]
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


class ControllerArgs(BaseModel):
    config_file_path: str
    num_gpus: int


def app_builder(args: ControllerArgs) -> Application:
    return ModelController.bind(args.config_file_path, args.num_gpus)
