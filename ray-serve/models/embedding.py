from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from logging import getLogger, Logger
from model_app import ModelAppInterface, ModelAppArgs
import psutil
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import socket
import subprocess
import time
from typing import Any


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

    def __init__(self, model_name: str, controller: str, gpus_per_replica: int) -> None:
        self._controller_app: DeploymentHandle = serve.get_app_handle(controller)
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
        await self._controller_app.handle_unavailable_model.remote(self._model_name)
        return False

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._is_active: bool = config["is_active"]
        if self._is_active:
            self._find_available_port()
            command = [
                "apptainer",
                "run",
                "--writable-tmpfs",
                "--nv",
                "docker://ghcr.io/huggingface/text-embeddings-inference:1.1",
                "--model-id",
                "BAAI/bge-large-en-v1.5",
                "--hostname",
                "0.0.0.0",
                "-p",
                f"{self._port}",
            ]
            # Start the command in the background
            process = subprocess.Popen(command)
            time.sleep(3)
            child_pids = []
            for proc in psutil.process_iter(attrs=["pid", "ppid"]):
                if proc.ppid() == process.pid:
                    child_pids.append(proc.pid)
            self._logger.info(
                f"Embedding model {self._model_name} launched, pid: {process.pid}, child_pids: {child_pids}"
            )
        else:
            self._logger.info(f"Embedding model {self._model_name} inactive")

    @app.post("/")
    def call(self, request: Request) -> Any:
        self._last_served_time = time.time()
        if not self._check_model_availability():
            return Response(status_code=503, content="Model not available")
        return {"embedding": [0.1, 0.2, 0.3]}

    def collect_eviction_defense_metrics(self) -> dict[str, Any]:
        return {"last_served_time": self._last_served_time}


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.controller, args.gpus_per_replica)
