from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
from logging import getLogger, Logger
from model_app import ModelAppInterface, ModelAppArgs
import psutil
from pydantic import BaseModel, ConfigDict
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import socket
import subprocess
import time
from typing import Any


class _hfEmbeddingReq(BaseModel):
    model: str
    model_config = ConfigDict(extra="allow")


app = FastAPI()


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):
    """
    This is a thin wrapper for Huggingface Embedding model.
    Refer to https://github.com/huggingface/text-embeddings-inference for details.
    The recommended way to deploy embedding model is to use docker container.
    In our HPC environment, we use apptainer to launch the container.
    The ray serve deployment is simply used to acquire resources e.g., setting CUDA_VISIBLE_DEVICES,
    and to provide a fastapi interface for the model.

    We deploy the container as an independent process. Ray is able to terminate user-spawned
    processes when the worker who created those processes exits.
    Refer to User-Spawned Processes in Ray document for details.
    """

    def __init__(self, model_name: str, main: str, gpus_per_replica: int) -> None:
        self._main: DeploymentHandle = serve.get_app_handle(main)
        self._is_active: bool = False
        self._logger: Logger = getLogger("ray.serve")
        self._model_name: str = model_name
        self._port: int
        self._gpus_per_replica: int = gpus_per_replica
        self._last_served_time: float = time.time()

    def _find_available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            self._port = s.getsockname()[1]
            self._logger.info(
                f"Found available port: {self._port} for embedding model."
            )
            return self._port

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            return True
        await self._main.handle_unavailable_model.remote(self._model_name)
        return False

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._is_active = config["is_active"]
        if self._is_active:
            self._find_available_port()
            command = [
                "apptainer",
                "run",
                "--writable-tmpfs",
                "--nv",
                "docker://ghcr.io/huggingface/text-embeddings-inference:1.1",
                "--model-id",
                f"{self._model_name}",
                "--hostname",
                "0.0.0.0",
                "-p",
                f"{self._port}",
            ]
            # Start the command in the background
            process = subprocess.Popen(command)

            # TODO: use program output to check the model status
            time.sleep(10)  # Wait for the model to be ready
            child_pids = []
            for proc in psutil.process_iter(attrs=["pid", "ppid"]):
                if proc.ppid() == process.pid:
                    child_pids.append(proc.pid)
            self._logger.info(
                f"Embedding model {self._model_name} launched, pid: {process.pid}, child_pids: {child_pids}"
            )
        else:
            self._logger.info(f"Embedding model {self._model_name} inactive")

    @app.get("/{full_path:path}")
    @app.post("/{full_path:path}")
    async def user_request(self, full_path: str, request: Request) -> Response:
        self._last_served_time = time.time()
        if not await self._check_model_availability():
            return Response(status_code=503, content="Embedding Model not available")

        # Forward the user request to the embedding model server
        target_url = f"http://localhost:{self._port}/{full_path}"
        method = request.method
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length", "content-type"]
        }  # Extract headers from the original request.

        async with httpx.AsyncClient() as client:
            if method.upper() == "GET":
                response = await client.request(method, target_url, headers=headers)

            elif method.upper() == "POST":
                body = await request.json()
                response = await client.request(
                    method, target_url, headers=headers, json=body
                )

            else:
                return Response(
                    status_code=405, content="Method not allowed", headers={}
                )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    async def create_request(
        self, request: _hfEmbeddingReq, http_method: str, full_path: str
    ) -> tuple[bool, Any]:
        self._last_served_time = time.time()
        if not await self._check_model_availability():
            return False, None

        # Forward the user request to the embedding model server
        target_url = f"http://localhost:{self._port}/{full_path}"

        async with httpx.AsyncClient() as client:
            if http_method.upper() == "GET":
                response = await client.request(http_method, target_url)
                content = response.json()

            elif http_method.upper() == "POST":
                response = await client.request(
                    http_method, target_url, json=request.model_dump()
                )
                content = response.json()

            else:
                content = "method not allowed."

            return True, content

    def collect_eviction_defense_metrics(self) -> dict[str, Any]:
        return {"last_served_time": self._last_served_time}


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.main, args.gpus_per_replica)
