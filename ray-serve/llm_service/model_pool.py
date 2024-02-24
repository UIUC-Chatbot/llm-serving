import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, Response
import json
from logging import getLogger, Logger
from pydantic import BaseModel
import ray
from ray import serve
from ray.serve import Application
from ray.serve.exceptions import RayServeException
from typing import AsyncGenerator
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


from config_writer import ConfigWriter
from model_context import ModelContext, ModelStatus, ModelType

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.
    It updates serve apps by modifying a config file and sending it to the Ray dashboard service.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    They have their own endpoints. ModelApp is an abstract class, and the actual model
    implementation is specified by the model_type.

    When a ModelApp is inactive but wants to load itself into GPUs, it sends a request to the
    ModelController to load the model. The ModelController might select victim models to evict from
    GPUs and load the requested model.

    A model daemon is deployed on the head node. It periodically checks the health of the model pool
    and the availability of GPUs. If a ModelApp becomes unhealthy, probably due to worker node it is
    running on being down, Ray will try to redeploy the ModelApp on other worker nodes. If, however,
    there is no available resources, then we need to remove those unhealthy ModelApps.
"""

"""
Mental Model:

    The ModelController is a centralized server that manages the model pool. It takes requests from
    users and executes them one at a time. User requests are essentially cooperative coroutines that
    are running in an event loop. Therefore, only one active coroutine is running at a time and it
    yields control to the event loop when it's "await"ing for something.

    Refer to https://docs.ray.io/en/latest/serve/architecture.html#lifetime-of-a-request for more
    details.

    When passing objects between Serve apps, we are actually passing copies of them. Under the hood,
    DeploymentResponse corresponds to a Ray ObjectRef, which is a reference to an immutable
    object in the object store. However, the response we obtain by "await"ing the remote call is a
    copy of the original object in the object store. Therefore, we can't modify the original object
    by modifying the response object.
"""


class _AdminRequest(BaseModel):
    key: str
    mode: str
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    model_type: str = "vllm_openai"
    gpus_per_replica: int = 1
    force: bool = False


main_app = FastAPI()


@serve.deployment(
    name="ModelController",
    ray_actor_options={"num_cpus": 1, "resources": {"head_agents": 1}},
    logging_config={"enable_access_log": False},
)
@serve.ingress(main_app)
class ModelController:
    """
    This class must be deployed on the head node, since it needs to send config file to Ray
    dashboard service, which might not be accessible from worker nodes.
    """

    def __init__(self, config_file_path: str, model_reference_path: str | None) -> None:
        self._config_writer: ConfigWriter = ConfigWriter(config_file_path)
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, ModelContext] = {}  # Currently registered models
        self._model_unsupported: dict[str, str] = {}  # Unsupported models
        if model_reference_path is not None:  # Number of GPUs required for large models
            try:
                with open(model_reference_path, "r") as f:
                    self._model_reference: dict = json.load(f)
                self._logger.info(f"{model_reference_path} successfully loaded.")
            except FileNotFoundError:
                self._logger.warning(f"{model_reference_path} not found.")
                self._model_reference: dict = {}
        else:
            self._model_reference: dict = {}
        self._num_gpus: int = int(ray.cluster_resources().get("GPU", 0))
        self._logger.info(f"ModelController initialized with {self._num_gpus} GPUs.")
        self._num_served_models: int = 0

        """
        The modification of the model pool should be atomic.
        If some coroutines need to modify the model pool and execute await statements during the
        modification process (yield control to the event loop), they must acquire the lock.
        """
        self._lock: asyncio.Lock = asyncio.Lock()

        """
        Apply the initial config, just in case the ModelController is restarted during service for
        some reason, it's better to restart the whole LLM service, since the model_pool is reset to
        empty and all past information is lost.
        (assuming the provided config file only has the default configs and no model apps configs)
        """
        self._config_writer.apply_config()
        self._logger.info("LLM Service initialized.")

    async def update_num_gpus(self) -> int:
        async with self._lock:
            self._num_gpus = int(ray.cluster_resources().get("GPU", 0))
        self._logger.info(f"Number of GPUs set to {self._num_gpus}.")
        return self._num_gpus

    def _count_available_gpus(self) -> int:
        """
        Return the number of available GPUs.
        """
        used_gpus: int = 0
        for model in self._model_pool.values():
            used_gpus += model.num_active_replicas * model.gpus_per_replica
        return self._num_gpus - used_gpus

    async def _get_healthy_model(self, model: ModelContext) -> ModelContext | None:
        """
        Check if the model is healthy. If the model is unhealthy, remove it from the model pool.
        Return model context if the model is healthy, None otherwise.
        This function should be called before calling a model app handle.
        """
        model_status: ModelStatus = await model.get_cached_status()
        match model_status:
            case ModelStatus.RUNNING:
                return model
            case ModelStatus.DEPLOY_FAILED:
                async with self._lock:
                    # Check whether the app exists because someone else might have removed it.
                    if model.model_name not in self._model_pool:
                        return None
                    if model.app_name != self._model_pool[model.model_name].app_name:
                        """
                        In rare cases, the app corresponding to the current model context has been
                        removed, but a new app with the same model name has been created, therefore
                        the model name can be found in the model pool, but the app name is
                        different.
                        """
                        return None

                    # If the deployment fails, then it is very likely something went wrong in the
                    # initialization function, so we should add this model to unsupported list.
                    self._model_unsupported[model.model_name] = model.error_msg
                    self._logger.warning(f"App {model.app_name} deployment failed.")
                    self._config_writer.remove_app(model)
                    self._model_pool.pop(model.model_name)
                    return None
            case _:
                async with self._lock:
                    # Check whether the app exists because someone else might have removed it.
                    if model.model_name not in self._model_pool:
                        return None
                    if model.app_name != self._model_pool[model.model_name].app_name:
                        """
                        In rare cases, the app corresponding to the current model context has been
                        removed, but a new app with the same model name has been created, therefore
                        the model name can be found in the model pool, but the app name is
                        different.
                        """
                        return None

                    self._logger.warning(f"App {model.app_name} is {model_status}.")
                    self._config_writer.remove_app(model)
                    self._model_pool.pop(model.model_name)
                    return None

    async def get_or_register_model(
        self,
        model_name: str,
        model_type: ModelType,
        gpus_per_replica: int = 1,
        force: bool = False,
    ) -> ModelContext | None:
        """
        Return a healthy model_context of the requested model. Create a serve app for the model
        if it's not created yet.
        Return None if model deployment is unhealthy.
        """
        if model_name in self._model_pool:
            model = await self._get_healthy_model(self._model_pool[model_name])
            if model is not None:
                if model.num_active_replicas == 0 and force:
                    await self.handle_unavailable_model(model.model_name)
                return model
            else:
                return None

        if model_name in self._model_unsupported:
            return None

        if gpus_per_replica > self._num_gpus:
            self._model_unsupported[model_name] = (
                f"Insufficient GPU resources for model {model_name} which requires {gpus_per_replica} GPUs."
            )
            return None

        # Create a new serve app for the requested model
        async with self._lock:
            # Check again because someone else might have added the model before we woke up.
            if model_name in self._model_unsupported:
                return None
            if model_name not in self._model_pool:
                model = ModelContext(
                    app_name=f"{model_name.replace('/', '--')}--{self._num_served_models}",
                    model_name=model_name,
                    route_prefix=f"/model-{self._num_served_models}",
                    model_type=model_type,
                    gpus_per_replica=gpus_per_replica,
                )
                model.status_reset()
                self._num_served_models += 1
                if force or self._count_available_gpus() >= gpus_per_replica:
                    model.num_active_replicas = 1
                    self._config_writer.add_app(model=model, is_active=True)
                else:
                    model.num_active_replicas = 0
                    self._config_writer.add_app(model=model, is_active=False)
                self._model_pool[model_name] = model

        return await self._get_healthy_model(model)

    async def delete_model_by_model_name(self, model_name: str) -> bool:
        """
        Delete the model app with the given model name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                self._config_writer.remove_app(self._model_pool[model_name])
                self._model_pool.pop(model_name)
                return True
            return False

    async def delete_model_by_app_name(self, app_name: str) -> bool:
        """
        Delete the model app with the given app name.
        """
        async with self._lock:
            for model in self._model_pool.values():
                if model.app_name == app_name:
                    self._config_writer.remove_app(model)
                    self._model_pool.pop(model.model_name)
                    return True
            return False

    async def reset_unsupported(self) -> None:
        """
        Reset the unsupported models.
        """
        async with self._lock:
            self._model_unsupported.clear()

    async def reset_all(self) -> None:
        """
        Reset LLM services.
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
            self._config_writer.deactivate_apps(models_out)
            for model_out in models_out:
                model_out.num_active_replicas = 0
                model_out.status_reset()
                asyncio.create_task(self._get_healthy_model(model_out))
        self._config_writer.activate_app(model_in)
        model_in.num_active_replicas = 1
        model_in.status_reset()
        asyncio.create_task(self._get_healthy_model(model_in))

    async def _select_victim(
        self, initiator: ModelContext, required_gpus: int, available_gpus: int
    ) -> list[ModelContext] | None:
        """
        The initiator model has requested to load itself into GPU. However, there is no available
        GPU. This function selects a victim model to evict from GPU.

        This function calls each model's collect_eviction_defense_metrics method to get the latest
        service information.
        """

        async def get_candidate(
            model: ModelContext,
        ) -> tuple[ModelContext, float] | None:
            try:
                """
                The caller of this function owns the lock, so we don't want this function to block
                for too long. We set a timeout and ignore the model if it doesn't respond in time.
                """
                metrics = await asyncio.wait_for(
                    model.app_handle.collect_eviction_defense_metrics.remote(),
                    timeout=10,
                )
                return model, metrics["last_served_time"]
            except (
                AttributeError,
                RayServeException,
                TimeoutError,
                asyncio.TimeoutError,
            ):
                return None

        candidates = []
        for model in self._model_pool.values():
            if model.app_name == initiator.app_name:
                continue
            if model.num_active_replicas == 0:
                continue
            candidates.append(get_candidate(model))

        candidate_reports = await asyncio.gather(*candidates)
        available_candidates: list[tuple[ModelContext, float]] = [
            candidate for candidate in candidate_reports if candidate is not None
        ]

        # Remove the least recently used model
        available_candidates.sort(key=lambda x: x[1])
        num_gpus_to_release = 0
        victims: list[ModelContext] = []
        for candidate in available_candidates:
            victims.append(candidate[0])
            num_gpus_to_release += (
                candidate[0].num_active_replicas * candidate[0].gpus_per_replica
            )
            if num_gpus_to_release + available_gpus >= required_gpus:
                break

        # TODO: do we need to wait here for a while? What if the victims don't fulfill the requirement?
        if num_gpus_to_release + available_gpus < required_gpus:
            return None
        else:
            return victims

    async def handle_unavailable_model(self, model_name: str) -> None:
        """
        This function is called by an inactive ModelApp who wants to load itself into GPU.

        It tries its best to unload some models to make room for the requested model, but there is
        no guarantee that the requested model will be loaded into GPU.
        """
        self._logger.info(f"Trying to load model {model_name} into GPUs.")
        model = self._model_pool[model_name]
        if model.num_active_replicas > 0:
            return  # If the model is already loaded, ignore this request.
        available_gpus = self._count_available_gpus()
        if available_gpus >= model.gpus_per_replica:
            return self._load_or_replace_model(model)

        # If there is no available GPU, evict models from GPUs and load the requested model.
        self._logger.info(
            f"Model {model_name} requires {model.gpus_per_replica - available_gpus} more GPUs, trying to evict other models."
        )
        async with self._lock:
            victims: list[ModelContext] | None = await self._select_victim(
                initiator=model,
                required_gpus=model.gpus_per_replica,
                available_gpus=available_gpus,
            )
            if victims is None:
                self._logger.info(f"Resource unavailable for model {model_name}.")
            else:
                self._load_or_replace_model(model, victims)

    def get_current_config(self) -> dict:
        return self._config_writer.get_current_config()

    """
    Admin API endpoints
    """

    @main_app.post("/admin")
    async def admin_call(self, request: _AdminRequest) -> str | dict:
        # TODO: the key is currently visible on GitHub. We need to change this.
        if request.key != "IloveRocknRoll":
            return "Permission denied. Aborting."

        match request.mode:
            case "get":
                if request.model_type == "vllm_raw":
                    model_type = ModelType.VLLM_RAW
                elif request.model_type == "vllm_openai":
                    model_type = ModelType.VLLM_OPENAI
                else:
                    return "Invalid model type. Aborting."
                model = await self.get_or_register_model(
                    model_name=request.model_name,
                    model_type=model_type,
                    gpus_per_replica=request.gpus_per_replica,
                    force=request.force,
                )
                if model is not None:
                    return f"Model {model.model_name} endpoint: {model.route_prefix}"
                else:
                    if request.model_name in self._model_unsupported:
                        return f"Model {request.model_name} not supported: {self._model_unsupported[request.model_name]}"
                    else:
                        return f"Model {request.model_name} initialization failed."

            case "delete":
                if await self.delete_model_by_model_name(request.model_name):
                    return f"Model {request.model_name} deleted."
                else:
                    return f"Model {request.model_name} not found."

            case "list":
                dump_model_pool = []
                for model in self._model_pool.values():
                    dump_model_pool.append(
                        {
                            "model_name": model.model_name,
                            "model_type": model.model_type,
                            "route_prefix": model.route_prefix,
                            "gpus_per_replica": model.gpus_per_replica,
                            "num_active_replicas": model.num_active_replicas,
                        }
                    )

                dump_model_unsupported = []
                for model_name, error_message in self._model_unsupported.items():
                    dump_model_unsupported.append(
                        {"model_name": model_name, "error_message": error_message}
                    )

                return {
                    "model_pool": dump_model_pool,
                    "model_unsupported": dump_model_unsupported,
                }

            case "dump_config":
                return self.get_current_config()

            case "info":
                return {
                    "num_total_gpus": self._num_gpus,
                    "num_available_gpus:": self._count_available_gpus(),
                    "num_served_models": self._num_served_models,
                }

            case "reset_unsupported":
                await self.reset_unsupported()
                return "Unsupported models reset."

            case "reset_all":
                await self.reset_all()
                return "LLM service reset."

            case _:
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
    async def create_chat_completion(self, request: ChatCompletionRequest):
        async def retry_func(func, num_retries):
            retry_count = 0
            while retry_count < num_retries:
                try:
                    return await func()
                except RayServeException:
                    retry_count += 1
                    await asyncio.sleep(0.1)
                except Exception as e:
                    raise e
            return JSONResponse(
                content="Service Temporarily Unavailable", status_code=503
            )

        async def create_batch_request(model: ModelContext) -> JSONResponse:
            is_success, response = await model.app_handle.options(
                stream=False
            ).create_chat_completion_batch.remote(request)

            if is_success:
                return JSONResponse(content=response)
            else:
                raise RayServeException("Model Not Available")

        async def create_stream_request(model: ModelContext) -> StreamingResponse:

            async def put_first_back(generator, first_item) -> AsyncGenerator:
                yield first_item
                async for item in generator:
                    yield item

            generator = model.app_handle.options(
                stream=True
            ).create_chat_completion_stream.remote(request)

            try:  # If the model is not available yet, it would return an empty generator
                first_item = await anext(generator)
            except:
                raise RayServeException("Model Not Available")

            # Since we have already consumed the first item, we need to put it back
            valid_generator = put_first_back(generator, first_item)

            return StreamingResponse(
                content=valid_generator, media_type="text/event-stream"
            )

        async def main_func():
            if request.model in self._model_reference:
                gpus_per_replica = self._model_reference[request.model]
            else:
                gpus_per_replica = 1
            model = await self.get_or_register_model(
                model_name=request.model,
                model_type=ModelType.VLLM_OPENAI,
                gpus_per_replica=gpus_per_replica,
            )
            if model is None:
                if request.model in self._model_unsupported:
                    return JSONResponse(
                        content=self._model_unsupported[request.model], status_code=400
                    )
                else:
                    return JSONResponse(content="Model Not Available", status_code=400)

            if request.stream:
                return await create_stream_request(model)
            else:
                return await create_batch_request(model)

        return await retry_func(main_func, 3)


class _ControllerArgs(BaseModel):
    config_file_path: str
    model_reference_path: str | None = None


def app_builder(args: _ControllerArgs) -> Application:
    return ModelController.bind(args.config_file_path, args.model_reference_path)
