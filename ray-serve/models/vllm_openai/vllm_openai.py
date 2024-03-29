# type: ignore
import argparse
import json
import os
import importlib
import inspect

import fastapi
from http import HTTPStatus
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ErrorResponse,
)
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRA

from model_app import ModelAppInterface, ModelAppArgs
from ray import serve
import time
from typing import AsyncGenerator


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split("=")
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


def parse_args(model_name, gpus_per_replica):
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="If provided, the server will require this key to be presented in the header.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs="+",
        action=LoRAParserAction,
        help="LoRA module configurations in the format name=path. Multiple modules can be specified.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The file path to the chat template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--response-role",
        type=str,
        default="assistant",
        help="The role name to return if " "`request.add_generation_prompt=true`.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="The file path to the SSL key file",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="The file path to the SSL cert file",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server using app.add_middleware(). ",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    # Important! Do not pass values from command line! This conflicts with Ray for whatever reason.
    # Pass values in the script and save you the headache.
    return parser.parse_args(
        [
            "--served-model-name",
            model_name,
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
    As of vllm 0.3.3, they only uses is_disconnected() function.
    """

    async def is_disconnected(self):
        return False


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):

    def __init__(self, model_name: str, main: str, gpus_per_replica: int) -> None:
        self.args = parse_args(model_name, gpus_per_replica)
        self.logger = init_logger(__name__)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.args.allowed_origins,
            allow_credentials=self.args.allow_credentials,
            allow_methods=self.args.allowed_methods,
            allow_headers=self.args.allowed_headers,
        )

        if token := os.environ.get("VLLM_API_KEY") or self.args.api_key:

            @app.middleware("http")
            async def authentication(request: Request, call_next):
                if not request.url.path.startswith("/v1"):
                    return await call_next(request)
                if request.headers.get("Authorization") != "Bearer " + token:
                    return JSONResponse(
                        content={"error": "Unauthorized"}, status_code=401
                    )
                return await call_next(request)

        for middleware in self.args.middleware:
            module_path, object_name = middleware.rsplit(".", 1)
            imported = getattr(importlib.import_module(module_path), object_name)
            if inspect.isclass(imported):
                app.add_middleware(imported)
            elif inspect.iscoroutinefunction(imported):
                app.middleware("http")(imported)
            else:
                raise ValueError(
                    f"Invalid middleware {middleware}. Must be a function or a class."
                )

        self.logger.info(f"vLLM API server version {vllm.__version__}")
        self.logger.info(f"args: {self.args}")

        if self.args.served_model_name is not None:
            self.served_model = self.args.served_model_name
        else:
            self.served_model = self.args.model

        # LLM-serving related fields
        self._is_active: bool = False
        self._main = serve.get_app_handle(main)
        self._last_served_time: float = time.time()
        self._unhandled_requests: int = 0

    def reconfigure(self, config) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config yaml file. Refer to Ray documentation for more details.
        """
        self._is_active: bool = config["is_active"]
        if self._is_active:
            engine_args = AsyncEngineArgs.from_cli_args(self.args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.openai_serving_chat = OpenAIServingChat(
                engine,
                self.served_model,
                self.args.response_role,
                self.args.lora_modules,
                self.args.chat_template,
            )
            self.openai_serving_completion = OpenAIServingCompletion(
                engine, self.served_model, self.args.lora_modules
            )

            app.root_path = self.args.root_path

    def collect_eviction_defense_metrics(self) -> dict:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        return {"last_served_time": self._last_served_time}

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            self._last_served_time = time.time()
            return True
        await self._main.handle_unavailable_model.remote(self.served_model)
        return False

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(self, _, exc):
        err = self.openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.get("/v1/models")
    async def show_available_models(self):
        models = await self.openai_serving_chat.show_available_models()
        return JSONResponse(content=models.model_dump())

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

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        if not await self._check_model_availability():
            return JSONResponse(
                content="Model Not Available, please try again later.", status_code=503
            )
        generator = await self.openai_serving_completion.create_completion(
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
