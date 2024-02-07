import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from logging import getLogger, Logger
from pydantic import BaseModel
import ray
from ray import serve
from ray.serve import Application
from ray.serve.exceptions import RayServeException
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest


from config_writer import ConfigWriter
from model_context import ModelContext, ModelType

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.
    It updates serve app by modifying the config file and sending it to the Ray dashboard service.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    They have their own endpoints. ModelApp is an abstract class, and the actual model
    implementation is specified by the model_type.

    When a ModelApp is inactive, users might request to load that model. The ModelApp sends a
    request to the ModelController to load the model. The ModelController might select a victim
    model to evict from GPU and load the requested model.
"""


class _AdminRequest(BaseModel):
    key: str
    mode: str
    model_name: str
    model_type: str = "vllm_openai"
    gpus_per_replica: int = 1
    config_dump_path: str = "latest_config.yaml"


class _FakeRequest:
    """
    The ModelController routes user requests to their corresponding ModelApp. However, VLLM OpenAI
    server uses starlette raw request object, which is not serializable. We need to create a fake
    request object, which is serializable, to pass to the model.
    As of vllm 0.3.0, they only uses is_disconnected() function.
    """

    async def is_disconnected(self):
        return False


main_app = FastAPI()


@serve.deployment
@serve.ingress(main_app)
class ModelController:
    def __init__(self, config_file_path: str, num_gpus: int) -> None:
        self._config_writer: ConfigWriter = ConfigWriter(config_file_path)
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, ModelContext] = {}  # Currently registered models
        self._model_unsupported: dict[str, str] = {}  # Unsupported models
        self._num_gpus: int = min(num_gpus, ray.cluster_resources().get("GPU", 0))
        self._num_served_models: int = 0

        # Lock required for modifying model_pool and model_unsupported, i.e., updating serve apps
        self._lock: asyncio.Lock = asyncio.Lock()

    def _has_available_gpu(self, required_gpus: int) -> bool:
        """
        Check if there is any available GPUs.
        """
        used_gpus: int = 0
        for model_name in self._model_pool.keys():
            used_gpus += self._model_pool[model_name].used_gpus
        return self._num_gpus >= used_gpus + required_gpus

    async def _check_model_health(self, model: ModelContext) -> bool:
        """
        Check if the model is healthy. If the model is unhealthy, remove it from the model pool.
        Return True if the model is healthy, False otherwise.
        This function should be called before calling a model app handle.
        """
        is_healthy = await model.check_health()
        if is_healthy:
            return True
        else:
            async with self._lock:
                if model.model_name in self._model_pool:
                    self._model_pool.pop(model.model_name)
                    self._model_unsupported[model.model_name] = (
                        model.get_error_message()
                    )
                    self._config_writer.remove_app(model)
            return False

    async def _get_or_register_model(
        self, model_name: str, model_type: ModelType, gpus_per_replica: int
    ) -> ModelContext | None:
        """
        Return the model_context of the requested model. Create a serve app for the model if it's
        not created yet.
        Return None if model deployment fails.
        """
        if model_name in self._model_pool:
            if await self._check_model_health(self._model_pool[model_name]):
                return self._model_pool[model_name]
            else:
                return None

        if model_name in self._model_unsupported:
            return None

        if gpus_per_replica > self._num_gpus:
            return None

        async with self._lock:
            if model_name in self._model_pool:
                return self._model_pool[model_name]
            if model_name in self._model_unsupported:
                return None

            # Create a new serve app for the requested model
            model_context = ModelContext(
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

            # Initialize the app handle and get the app status
            model_context.health_reset()
            if await model_context.check_health():
                self._model_pool[model_name] = model_context
                return model_context
            else:
                self._model_unsupported[model_name] = model_context.get_error_message()
                self._config_writer.remove_app(model=model_context)
                return None

    async def _delete_model(self, model_name: str) -> bool:
        """
        Delete the model app with the given name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                self._config_writer.remove_app(model=self._model_pool[model_name])
                self._model_pool.pop(model_name)
                return True
            else:
                return False

    async def _reset(self) -> None:
        """
        Delete all model apps.
        """
        async with self._lock:
            all_models = [model for model in self._model_pool.values()]
            self._config_writer.remove_apps(all_models)
            self._model_pool.clear()
            self._model_unsupported.clear()
            self._logger.info("LLM service reset.")

    def _load_or_replace_model(
        self, model_in: ModelContext, models_out: None | list[ModelContext] = None
    ) -> None:
        """
        Loads model_in into GPU. Unloads model_out from GPU if necessary.
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify whether deactivating model_out actually releases GPU
        resources, nor does it check for the availability of any GPU.
        """
        if models_out is not None:
            for model_out in models_out:
                self._config_writer.deactivate_app(model_out)
                model_out.used_gpus = 0
                model_out.health_reset()
        self._config_writer.activate_app(model_in)
        model_in.used_gpus = model_in.gpus_per_replica
        model_in.health_reset()

    async def _select_victim(
        self, initiator: ModelContext, required_gpus: int
    ) -> list[ModelContext]:
        """
        The initiator model has requested to load itself into GPU. However, there is no available
        GPU. This function selects a victim model to evict from GPU.
        """
        candidates = []
        victims: list[ModelContext] = []
        for model in self._model_pool.values():
            if model.app_name == initiator.app_name:
                continue
            try:
                metrics = (
                    await model.app_handle.collect_eviction_defense_metrics.remote()
                )
            except RayServeException:
                continue
            candidates.append((model, metrics["last_served_time"]))

        # Remove the least recently used model
        candidates.sort(key=lambda x: x[1])
        num_gpus_to_release = 0
        for candidate in candidates:
            victims.append(candidate[0])
            num_gpus_to_release += candidate[0].used_gpus
            if num_gpus_to_release >= required_gpus:
                break
        # TODO: do we need to wait here for a while? What if the victims don't fulfill the requirement?
        return victims

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
            victims: list[ModelContext] = await self._select_victim(
                model, model.gpus_per_replica
            )
            return self._load_or_replace_model(model, victims)

    """
    Admin API endpoints
    """

    @main_app.post("/admin")
    async def admin_call(self, request: _AdminRequest) -> str:
        # TODO: the key is currently visible on GitHub. We need to change this.
        if request.key != "IloveRocknRoll":
            return "Permission denied. Aborting."
        if request.mode == "get":
            if request.model_type == "vllm_raw":
                model_type = ModelType.VLLM_RAW
            elif request.model_type == "vllm_openai":
                model_type = ModelType.VLLM_OPENAI
            else:
                return "Invalid model type. Aborting."
            model_context = await self._get_or_register_model(
                model_name=request.model_name,
                model_type=model_type,
                gpus_per_replica=request.gpus_per_replica,
            )
            if model_context is not None:
                return f"Model {model_context.model_name} endpoint: {model_context.route_prefix}"
            else:
                return f"Model {request.model_name} not supported: {self._model_unsupported[request.model_name]}"
        elif request.mode == "delete":
            if await self._delete_model(request.model_name):
                return f"Model {request.model_name} deleted."
            else:
                return f"Model {request.model_name} not found."
        elif request.mode == "dump_config":
            self._config_writer.dump_config(request.config_dump_path)
            return f"Config dumped to {request.config_dump_path}"
        elif request.mode == "reset":
            await self._reset()
            return "LLM service reset."
        else:
            return "Invalid mode. Aborting."

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
        retry = 0
        while retry < 3:
            try:
                model_context = await self._get_or_register_model(
                    model_name=request.model,
                    model_type=ModelType.VLLM_OPENAI,
                    gpus_per_replica=1,
                )
                if model_context is None:
                    return JSONResponse(
                        content=self._model_unsupported[request.model], status_code=400
                    )

                response = await model_context.app_handle.create_chat_completion.remote(
                    request, _FakeRequest()
                )
                return response
            except RayServeException:
                retry += 1
                await asyncio.sleep(0.3)
        raise RayServeException("Service unavailable. Please try again later.")

    @main_app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        retry = 0
        while retry < 3:
            try:
                model_context = await self._get_or_register_model(
                    model_name=request.model,
                    model_type=ModelType.VLLM_OPENAI,
                    gpus_per_replica=1,
                )
                if model_context is None:
                    return JSONResponse(
                        content=self._model_unsupported[request.model], status_code=400
                    )

                response = await model_context.app_handle.create_completion.remote(
                    request, _FakeRequest()
                )
                return response
            except RayServeException:
                retry += 1
                await asyncio.sleep(0.3)
        raise RayServeException("Service unavailable. Please try again later.")


class _ControllerArgs(BaseModel):
    config_file_path: str
    num_gpus: int


def app_builder(args: _ControllerArgs) -> Application:
    return ModelController.bind(args.config_file_path, args.num_gpus)
