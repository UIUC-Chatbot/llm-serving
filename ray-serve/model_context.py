import asyncio
from enum import Enum
from ray import serve
from ray.serve.handle import DeploymentHandle
import time


class ModelType(Enum):
    VLLM_RAW = 1  # Raw VLLM model, created by llm = LLM(model="model_name")
    VLLM_OPENAI = 2  # VLLM OpenAI-Compatible server


class ModelContext:
    def __init__(
        self,
        app_name: str,
        model_name: str,
        route_prefix: str,
        model_type: ModelType,
        gpus_per_replica: int,
    ) -> None:
        self.app_name: str = app_name
        self.app_handle: DeploymentHandle
        self.model_name: str = model_name
        self.model_type: ModelType = model_type
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment
        self.created_time: int = int(time.time())
        self.gpus_per_replica: int = gpus_per_replica
        self.used_gpus: int = 0
        self._is_healthy: bool = False
        self._health_checked: bool = False

    def health_reset(self) -> None:
        self._is_healthy = False
        self._health_checked = False

    async def check_health(self) -> bool:
        """
        Get the app handle to the serve app and check for its health.
        Note that app updates via config file are async, so we might need to wait for a while before
        the app is actually created.
        """
        if self._health_checked:
            return self._is_healthy

        while True:
            app_status = serve.status().applications[self.app_name].status
            if app_status == "RUNNING":
                self.app_handle = serve.get_app_handle(self.app_name)
                self._is_healthy = True
                break
            elif app_status == "DEPLOY_FAILED" or app_status == "DELETING":
                self._is_healthy = False
                break
            await asyncio.sleep(1)
        self._health_checked = True
        return self._is_healthy

    def get_error_message(self) -> str:
        msg = (
            serve.status()
            .applications[self.app_name]
            .deployments[self.wrapper_name]
            .message
        )
        return msg
