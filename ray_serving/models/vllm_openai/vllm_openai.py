# type: ignore
import fastapi
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus

from logging import getLogger, Logger
from model_app import ModelAppInterface, ModelAppArgs
from ray import serve
import time
from typing import AsyncGenerator

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.usage.usage_lib import UsageContext


def parse_args(model_name: str, gpus_per_replica: int):
    """
    Important! Do not pass values from command line! This conflicts with Ray for whatever reason.
    Pass values in the script and save you the headache.
    """
    parser = make_arg_parser()
    return parser.parse_args(
        [
            "--model",
            model_name,
            "--trust-remote-code",
            "--tensor-parallel-size",
            f"{gpus_per_replica}",
            "--device",
            "cuda",
            "--enforce-eager",
        ]
    )


app = fastapi.FastAPI()


class _FakeRequest:
    """
    VLLM OpenAI server uses starlette raw request object, which is not serializable. We need to
    create a fake request object, which is serializable, to pass to the model.
    As of vllm 0.4.0, they only uses is_disconnected() function.
    """

    async def is_disconnected(self):
        return False


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):

    def __init__(self, model_name: str, main: str, gpus_per_replica: int) -> None:
        self._args = parse_args(model_name, gpus_per_replica)
        self._logger: Logger = getLogger("ray.serve")

        self._logger.info(f"vLLM API server version {vllm.__version__}")
        self._logger.info(f"args: {self._args}")

        if self._args.served_model_name is not None:
            self._served_model = self._args.served_model_name
        else:
            self._served_model = self._args.model

        # LLM-serving related fields
        self._is_active: bool = False
        self._main = serve.get_app_handle(main)
        self._last_served_time: float = time.time()

    def reconfigure(self, config) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config yaml file. Refer to Ray documentation for more details.
        """
        self._is_active: bool = config["is_active"]
        if self._is_active:
            engine_args = AsyncEngineArgs.from_cli_args(self._args)
            engine = AsyncLLMEngine.from_engine_args(
                engine_args, usage_context=UsageContext.OPENAI_API_SERVER
            )
            self.openai_serving_chat: OpenAIServingChat = OpenAIServingChat(
                engine,
                self._served_model,
                self._args.response_role,
                self._args.lora_modules,
                self._args.chat_template,
            )
            self.openai_serving_completion: OpenAIServingCompletion = (
                OpenAIServingCompletion(
                    engine, self._served_model, self._args.lora_modules
                )
            )

    def collect_eviction_defense_metrics(self) -> dict:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        return {"last_served_time": self._last_served_time}

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            self._last_served_time = time.time()
            return True
        await self._main.handle_unavailable_model.remote(self._served_model)
        return False

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(self, _, exc):
        err = self.openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        await self.openai_serving_chat.engine.check_health()
        return Response(status_code=200)

    @app.get("/v1/models")
    async def show_available_models(self):
        models = await self.openai_serving_chat.show_available_models()
        return JSONResponse(content=models.model_dump())

    @app.get("/version")
    async def show_version(self):
        ver = {"version": vllm.__version__}
        return JSONResponse(content=ver)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if not await self._check_model_availability():
            return JSONResponse(
                content="Model Not Available, please try again later.", status_code=503
            )

        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )

        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.model_dump())

    async def create_chat_completion_batch(
        self, request: ChatCompletionRequest
    ) -> tuple[bool, dict] | tuple[bool, None]:
        """
        For internal use only. This method is called by the controller to get the model response
        in batch.
        """

        if not await self._check_model_availability():
            return False, None

        response = await self.openai_serving_chat.create_chat_completion(
            request, _FakeRequest()
        )
        return True, response.model_dump()

    async def create_chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        """
        For internal use only. This method is called by the controller to get the model response
        in stream.

        The original method returns a StreamingResponse wrapping an async generator. However, this
        response is not serializable and cannot be returned directly to the controller in Ray.

        Ray supports streaming response by returning generators, so we have to make this function
        itself a generator.

        Refer to https://docs.ray.io/en/latest/serve/tutorials/streaming.html for more details.
        """

        if not await self._check_model_availability():
            return

        response = await self.openai_serving_chat.create_chat_completion(
            request, _FakeRequest()
        )
        async for chunk in response:
            yield chunk


def app_builder(args: ModelAppArgs):
    return ModelApp.bind(args.model_name, args.main, args.gpus_per_replica)
