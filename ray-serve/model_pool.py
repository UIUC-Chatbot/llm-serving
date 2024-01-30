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
from starlette.requests import Request
import yaml

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models on demand and performs model switching when necessary. It also handles model deletion requests.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.

    When a ModelApp is inactive, it might request the ModelController to load the model. The ModelController might select a victim model to evict from GPU and load the requested model.
"""

main_app = fastapi.FastAPI()


class _AppAction(Enum):
    ADD = 1  # Add a new serve app
    REMOVE = 2  # Remove a serve app
    UPDATE = 3  # Update a serve app


class _DeploymentAction(Enum):
    ACTIVATE = 1  # Activate a deployment
    DEACTIVATE = 2  # Deactivate a deployment


class _ModelContext:
    def __init__(self, app_name: str, model_name: str, route_prefix: str) -> None:
        self.app_name: str = app_name
        self.app_handle: DeploymentHandle | None = None
        self.model_name: str = model_name
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the deployment
        self.used_gpus: int = 0


class UserRequest(BaseModel):
    mode: str
    model_name: str
    file_path: str | None = None


@serve.deployment
@serve.ingress(main_app)
class ModelController:
    def __init__(self, config_file_path: str, num_gpus: int) -> None:
        with open(config_file_path, "r") as file:
            self._config = yaml.safe_load(file)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, _ModelContext] = {}
        self._num_gpus: int = num_gpus
        self._num_served_models: int = 0

    def _has_available_gpu(self) -> bool:
        """
        Check if there is any available GPU.
        """
        used_gpus: int = 0
        for model_name in self._model_pool.keys():
            used_gpus += self._model_pool[model_name].used_gpus
        return self._num_gpus > used_gpus

    def _update_app(
        self,
        app_action: _AppAction,
        model: _ModelContext,
        deploy_action: _DeploymentAction | None = None,
    ) -> None:
        """
        Ray uses config file (yaml) to dynamically update serve apps and deployments.
        ModelController maintains a local copy of the config file, which is updated whenever a model is added, removed, activated, or deactivated.
        ModelController sends the config file to the Ray dashboard service, which then updates the serve apps and deployments.
        """
        apps: list[dict] = self._config["applications"]
        if app_action == _AppAction.ADD:
            if self._has_available_gpu():
                num_gpus: int = 1
                is_active: bool = True
                model.used_gpus = 1
            else:
                num_gpus = 0
                is_active = False
                model.used_gpus = 0
            apps.append(
                {
                    "name": model.app_name,
                    "route_prefix": model.route_prefix,
                    "import_path": "model_app:app_builder",
                    "args": {
                        "model_name": model.model_name,
                        "controller": self._config["applications"][0]["name"],
                    },
                    "deployments": [
                        {
                            "name": model.wrapper_name,
                            "num_replicas": 1,
                            "ray_actor_options": {"num_gpus": num_gpus},
                            "user_config": {"is_active": is_active},
                        },
                    ],
                }
            )
        elif app_action == _AppAction.REMOVE:
            self._config["applications"] = [
                app for app in apps if app.get("name") != model.app_name
            ]
        elif app_action == _AppAction.UPDATE:
            app = next(
                (d for d in apps if d.get("name") == model.app_name),
                None,
            )
            if deploy_action == _DeploymentAction.ACTIVATE:
                app["deployments"][0]["ray_actor_options"]["num_gpus"] = 1
                app["deployments"][0]["user_config"]["is_active"] = True
                model.used_gpus = 1
            else:
                app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0
                app["deployments"][0]["user_config"]["is_active"] = False
                model.used_gpus = 0

        # The following code sends a request containing the updated config file to the Ray
        # dashboard, which then updates the serve apps and deployments. The call is async; it
        # returns immediately without waiting for the dashboard to finish the update.
        ServeDeploySchema.parse_obj(self._config)
        address: str = os.environ.get("RAY_DASHBOARD_ADDRESS", "http://localhost:8265")
        ServeSubmissionClient(address).deploy_applications(self._config)

        # Log the update
        deploy_status: str = f", Deployment: {deploy_action}" if deploy_action else ""
        self._logger.info(
            f"Application updated.  App: {model.app_name}, Action: {app_action}{deploy_status}"
        )

    async def _get_or_register_model(self, model_name: str) -> str:
        """
        Return the route prefix of the requested model. Create a serve app for the model if it's not created yet.
        """
        async with self._lock:
            if model_name not in self._model_pool:
                model_context = _ModelContext(
                    app_name=model_name.replace("/", "---"),
                    model_name=model_name,
                    route_prefix=f"/model-{self._num_served_models}",
                )
                self._model_pool[model_name] = model_context
                self._num_served_models += 1
                self._update_app(_AppAction.ADD, model_context)
            return self._model_pool[model_name].route_prefix

    async def _delete_model(self, model_name: str) -> str:
        """
        Delete the model app with the given name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                self._update_app(_AppAction.REMOVE, self._model_pool[model_name])
                self._model_pool.pop(model_name)
                return f"Model {model_name} deleted."
            else:
                return f"Model {model_name} not found."

    @main_app.post("/")
    async def call(self, request: UserRequest) -> str:
        """
        The entrypoint of the ModelController. The input should be in json format, containing the following fields:
            "mode": "get", "delete", "dump_config"
            "model_name": the model name as recognized by VLLM or HuggingFace
        """
        mode: str = request.mode
        model_name: str = request.model_name

        if mode == "get":
            return await self._get_or_register_model(model_name)
        elif mode == "delete":
            return await self._delete_model(model_name)
        elif mode == "dump_config":
            file_path: str = request.file_path
            with open(file_path, "w") as file:
                yaml.dump(self._config, file, sort_keys=False)
            return f"Config dumped to {file_path}"
        else:
            return "Invalid mode. Aborting."

    def _load_or_replace_model(
        self, model_in: _ModelContext, model_out: None | _ModelContext = None
    ) -> None:
        """
        Loads model_in into GPU. Unloads model_out from GPU if necessary.
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify whether deactivating model_out actually releases GPU resources, nor does it check for the availability of any GPU.
        """
        if model_out is not None:
            self._update_app(_AppAction.UPDATE, model_out, _DeploymentAction.DEACTIVATE)
        self._update_app(_AppAction.UPDATE, model_in, _DeploymentAction.ACTIVATE)

    async def _select_victim(self, initiator: _ModelContext) -> _ModelContext:
        candidates = []
        for model in self._model_pool.values():
            if model.app_name == initiator.app_name:
                continue
            if model.used_gpus == 0:
                continue
            if model.app_handle is None:
                model.app_handle = serve.get_app_handle(model.app_name)
            metrics = await model.app_handle.collect_eviction_defense_metrics.remote()
            candidates.append((model, metrics["last_served_time"]))
        victim = min(candidates, key=lambda x: x[1])[0]
        # TODO: do we need to wait here for a while?
        return victim

    async def handle_unavailable_model(self, model_name: str) -> None:
        """
        This function is called by a ModelApp that fails to load a model.
        """
        async with self._lock:
            model = self._model_pool[model_name]
            if model.used_gpus > 0:
                return  # If the model is already loaded, ignore this request.
            if self._has_available_gpu():
                # If there is an available GPU, load the model.
                return self._load_or_replace_model(model)
            # If there is no available GPU, evict a model from GPU and load the requested model.
            victim: _ModelContext = await self._select_victim(model)
            return self._load_or_replace_model(model, victim)


class ControllerArgs(BaseModel):
    config_file_path: str
    num_gpus: int


def app_builder(args: ControllerArgs) -> Application:
    return ModelController.bind(args.config_file_path, args.num_gpus)
