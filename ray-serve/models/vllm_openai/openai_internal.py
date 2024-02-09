# type: ignore
import argparse
import json

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import add_global_metrics_labels
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

from model_app import ModelAppInterface, ModelAppArgs
from ray import serve
import time
from typing import AsyncGenerator


def parse_args(model_name: str):
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
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
        ]
    )


class _FakeRequest:
    """
    VLLM OpenAI server uses starlette raw request object, which is not serializable. We need to
    create a fake request object, which is serializable, to pass to the model.
    As of vllm 0.3.0, they only uses is_disconnected() function.
    """

    async def is_disconnected(self):
        return False


@serve.deployment(name="ModelApp")
class ModelApp(ModelAppInterface):

    def __init__(self, model_name: str, controller: str) -> None:
        self.args = parse_args(model_name)
        self.logger = init_logger(__name__)
        self.logger.info(f"args: {self.args}")

        global served_model
        if self.args.served_model_name is not None:
            self.served_model = self.args.served_model_name
        else:
            self.served_model = self.args.model

        # LLM-serving related fields
        self._is_active: bool = False
        self._controller_app = serve.get_app_handle(controller)
        self._last_served_time = time.time()
        self._unhandled_requests = 0

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
                self.args.chat_template,
            )
            self.openai_serving_completion = OpenAIServingCompletion(
                engine, self.served_model
            )

            # Register labels for metrics
            add_global_metrics_labels(model_name=engine_args.model)

    def collect_eviction_defense_metrics(self) -> dict:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        return {"last_served_time": self._last_served_time}

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            self._last_served_time = time.time()
            return True
        await self._controller_app.handle_unavailable_model.remote(self.served_model)
        return False

    async def create_chat_completion_batch(
        self, request: ChatCompletionRequest
    ) -> dict:
        """
        Returns True if the request is successfully served, and the response.
        Returns False, possibly with an error message, if model is unavailable or inference
        has failed.
        """
        if not await self._check_model_availability():
            raise RuntimeError("Model Not Available")

        response = await self.openai_serving_chat.create_chat_completion(
            request, _FakeRequest()
        )

        if isinstance(response, ErrorResponse):
            raise RuntimeError(str(response.model_dump()))
        elif isinstance(response, ChatCompletionResponse):
            return response.model_dump()
        else:
            raise RuntimeError("Model Error")

    async def create_chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        """
        Returns True if the request is successfully served, and an asynchronous generator of the
        response.
        Returns False, possibly with an error message, if model is unavailable or inference
        has failed.
        """
        if not await self._check_model_availability():
            raise RuntimeError("Model Not Available")

        response = await self.openai_serving_chat.create_chat_completion(
            request, _FakeRequest()
        )

        if isinstance(response, ErrorResponse):
            raise RuntimeError(str(response.model_dump()))
        elif isinstance(response, ChatCompletionResponse):
            raise RuntimeError(str(response.model_dump()))
        else:
            async for chunk in response:
                yield chunk


def app_builder(args: ModelAppArgs):
    return ModelApp.bind(args.model_name, args.controller)
