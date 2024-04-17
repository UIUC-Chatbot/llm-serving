import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
import json
from pydantic import BaseModel, ConfigDict
from ray import serve
from ray.serve import Application
from ray.serve.exceptions import RayServeException
from typing import AsyncGenerator
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

from model_context import ModelContext, ModelType
from model_controller import ModelController


"""
This file is a wrapper around the ModelController class, which is responsible for providing the
fastapi endpoints.
"""


class _AdminGetModelReq(BaseModel):
    key: str
    model_name: str
    model_type: str
    gpus_per_replica: int
    hf_key: str | None = None
    force_load: bool = False


class _AdminDelModelReq(BaseModel):
    key: str
    model_name: str


class _AdminReq(BaseModel):
    key: str


class _AdminModelRefReq(BaseModel):
    key: str
    model_reference_path: str


class _hfEmbeddingReq(BaseModel):
    model: str
    model_config = ConfigDict(extra="allow")


main_app = FastAPI()


@serve.deployment(
    name="MainApp",
    ray_actor_options={"num_cpus": 1, "resources": {"head_agents": 1}},
    logging_config={"enable_access_log": False},
)
@serve.ingress(main_app)
class MainApp(ModelController):
    """
    This class must be deployed on the head node, since it needs to send config file to Ray
    dashboard service, which might not be accessible from worker nodes.
    """

    def __init__(
        self,
        config_file_path: str,
        dashboard_port: int,
        has_autoscaler: bool,
        model_reference_path: str | None,
    ) -> None:
        super().__init__(config_file_path, dashboard_port, has_autoscaler)
        if model_reference_path is not None:
            self.load_model_reference(model_reference_path)
        else:
            self._model_reference: dict = {}

    def load_model_reference(self, model_reference_path: str) -> None:
        """
        model_reference is a file that contains important information about the models, such as the
        number of GPUs required for each model and priority.
        """
        try:
            with open(model_reference_path, "r") as f:
                self._model_reference = json.load(f)
            self._logger.info(f"{model_reference_path} successfully loaded.")
        except FileNotFoundError:
            self._model_reference = {}
            self._logger.warning(f"{model_reference_path} not found.")

    @main_app.get("/health")
    def check_health(self) -> Response:
        """Health check."""
        return Response(status_code=200, content="LLM service is healthy.")

    @main_app.get("/models")
    def list_models(self) -> JSONResponse:
        hot_models = []
        cold_models = []
        for model in self._model_pool.values():
            if model.is_deployment_success is None:
                status = "Deploying"
            elif model.is_deployment_success:
                status = "Running"
            else:
                status = "Failed"

            model_info: dict = {
                "model_name": model.model_name,
                "model_type": str(model.model_type),
                "status": status,
                "priority": model.priority,
                "route_prefix": model.route_prefix,
                "gpus_per_replica": model.gpus_per_replica,
            }
            if model.num_active_replicas > 0:
                hot_models.append(model_info)
            else:
                cold_models.append(model_info)
        return JSONResponse(
            status_code=200,
            content={"hot_models": hot_models, "cold_models": cold_models},
        )

    """
    Admin API endpoints
    """

    def _verify_key(self, key: str) -> bool:
        # TODO: the key is currently visible on GitHub. We need to change this.
        return key == "IloveRocknRoll"

    @main_app.post("/admin/get_model")
    async def admin_get_model(self, request: _AdminGetModelReq) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")

        if request.model_type == "empty":
            model_type = ModelType.EMPTY
        elif request.model_type == "vllm_raw":
            model_type = ModelType.VLLM_RAW
        elif request.model_type == "vllm_openai":
            model_type = ModelType.VLLM_OPENAI
        elif request.model_type == "embedding":
            model_type = ModelType.EMBEDDING
        else:
            return JSONResponse(
                status_code=400, content="Invalid model type. Aborting."
            )

        if request.model_name in self._model_reference:
            priority: int = self._model_reference[request.model_name]["priority"]
            gpus_per_replica: int = self._model_reference[request.model_name][
                "gpus_per_replica"
            ]
        else:
            priority: int = 0
            gpus_per_replica: int = request.gpus_per_replica

        model = await self.get_or_register_model(
            model_name=request.model_name,
            model_type=model_type,
            priority=priority,
            gpus_per_replica=gpus_per_replica,
            hf_key=request.hf_key,
            force_load=request.force_load,
        )
        if model is not None:
            return JSONResponse(
                status_code=200,
                content=f"Model {model.model_name} endpoint: {model.route_prefix}",
            )
        else:
            return JSONResponse(
                status_code=503,
                content=f"Model {request.model_name} initialization failed",
            )

    @main_app.post("/admin/delete_model")
    async def admin_delete_model(self, request: _AdminDelModelReq) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")

        if await self.delete_model_by_model_name(request.model_name):
            return JSONResponse(
                status_code=200, content=f"Model {request.model_name} deleted."
            )
        else:
            return JSONResponse(
                status_code=400,
                content=f"Model {request.model_name} not found.",
            )

    @main_app.post("/admin/reset")
    async def admin_reset(self, request: _AdminReq) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")
        await self.reset_all()
        return JSONResponse(status_code=200, content="LLM service reset.")

    @main_app.post("/admin/info")
    async def admin_info(self, request: _AdminReq) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")
        service_info: dict = {
            "has_autoscaler": self._has_autoscaler,
            "last_autoscaler_exhaustion_time": self._last_exhaustion_time,
            "num_total_gpus": self._num_gpus,
            "num_available_gpus:": self.count_available_gpus(),
            "num_served_models": self._num_served_models,
        }
        return JSONResponse(status_code=200, content=service_info)

    @main_app.post("/admin/dump_config")
    async def admin_dump_config(self, request: _AdminReq) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")
        config_file = self.get_current_config()
        return JSONResponse(status_code=200, content=config_file)

    @main_app.post("/admin/load_model_reference")
    async def admin_load_model_reference(
        self, request: _AdminModelRefReq
    ) -> JSONResponse:
        if not self._verify_key(request.key):
            return JSONResponse(status_code=403, content="Permission denied. Aborting.")
        model_reference_path = request.model_reference_path
        self.load_model_reference(model_reference_path)
        return JSONResponse(
            status_code=200, content="Function load_model_reference() executed."
        )

    """
    A helper function for retrying a function.
    """

    async def _retry_func(self, func, num_retries):
        retry_count = 0
        while retry_count < num_retries:
            try:
                return await func()
            except RayServeException:
                retry_count += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                raise e
        return JSONResponse(content="Service Temporarily Unavailable", status_code=503)

    """
    OpenAI API endpoints
    """

    @main_app.get("/v1/models")
    async def list_openai_models(self):
        model_pool: dict[str, ModelContext] = self.get_model_pool()
        models = []
        for model in model_pool.values():
            if model.model_type == ModelType.VLLM_OPENAI:
                model_info: dict = {
                    "id": model.model_name,
                    "object": "model",
                    "created": model.created_time,
                    "owned_by": "NCSA",
                }
                models.append(model_info)
        return {"object": "list", "data": models}

    @main_app.post("/v1/chat/completions")
    async def create_openai_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):

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
                priority: int = self._model_reference[request.model]["priority"]
                gpus_per_replica: int = self._model_reference[request.model][
                    "gpus_per_replica"
                ]
            else:
                priority: int = 0
                gpus_per_replica = 1

            # Huggingface key authorization
            hf_key = None
            auth_header = raw_request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                if token.startswith("hf_"):
                    hf_key = token
                else:
                    self._logger.info("Received invalid huggingface key.")

            model = await self.get_or_register_model(
                model_name=request.model,
                model_type=ModelType.VLLM_OPENAI,
                priority=priority,
                gpus_per_replica=gpus_per_replica,
                hf_key=hf_key,
            )
            if model is None:
                return JSONResponse(content="Model is not supported.", status_code=400)

            if request.stream:
                return await create_stream_request(model)
            else:
                return await create_batch_request(model)

        return await self._retry_func(main_func, 2)

    """
    Huggingface Embedding API endpoints
    """

    @main_app.post("/hf_embed/{full_path:path}")
    async def create_hf_embedding(
        self, full_path: str, request: _hfEmbeddingReq, raw_request: Request
    ) -> JSONResponse:

        async def main_func():
            if request.model in self._model_reference:
                priority: int = self._model_reference[request.model]["priority"]
            else:
                priority: int = 0

            # Huggingface key authorization
            hf_key = None
            auth_header = raw_request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                if token.startswith("hf_"):
                    hf_key = token
                else:
                    self._logger.info("Received invalid huggingface key.")

            model = await self.get_or_register_model(
                model_name=request.model,
                model_type=ModelType.EMBEDDING,
                priority=priority,
                gpus_per_replica=1,  # Assume embedding model uses 1 GPU only
                hf_key=hf_key,
            )
            if model is None:
                return JSONResponse(content="Model is not supported.", status_code=400)

            is_success, response = await model.app_handle.create_request.remote(
                request, raw_request.method, full_path
            )

            if is_success:
                return JSONResponse(content=response)
            else:
                raise RayServeException("Model Not Available")

        return await self._retry_func(main_func, 2)


class _MainArgs(BaseModel):
    config_file_path: str
    dashboard_port: int
    has_autoscaler: bool = False
    model_reference_path: str | None = None


def app_builder(args: _MainArgs) -> Application:
    return MainApp.bind(
        args.config_file_path,
        args.dashboard_port,
        args.has_autoscaler,
        args.model_reference_path,
    )
