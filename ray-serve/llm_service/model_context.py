import asyncio
from enum import Enum
from ray import serve
from ray.serve.handle import DeploymentHandle
import time


class ModelStatus(Enum):
    RUNNING = 1  # The model is running and healthy
    DEPLOY_FAILED = 2  # The latest model deployment failed
    UNHEALTHY = 3  # The model is currently unhealthy after a successful deployment
    NONEXISTENT = 4  # The model does not exist


class ModelType(Enum):
    VLLM_RAW = 1  # Raw VLLM model, created by llm = LLM(model="model_name")
    VLLM_OPENAI = 2  # VLLM OpenAI-Compatible server


class ModelPath:
    @staticmethod
    def get_import_path(model_type: ModelType) -> str:
        """
        Return a path to the model implementation file.
        """
        if model_type == ModelType.VLLM_RAW:
            return "models.vllm_raw:app_builder"

        elif model_type == ModelType.VLLM_OPENAI:
            return "models.vllm_openai.vllm_openai:app_builder"

        else:
            raise ValueError("Invalid model type.")


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
        self.error_msg: str = ""  # error message, and possibly stack traces
        self.model_name: str = model_name
        self.model_type: ModelType = model_type
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment
        self.created_time: int = int(time.time())
        self.gpus_per_replica: int = gpus_per_replica
        self.num_active_replicas: int = 0
        self._deployment_status: ModelStatus | None = None

    def status_reset(self) -> None:
        self._deployment_status = None

    async def check_model_status(self) -> ModelStatus:
        """
        Note that app updates via config file are async, so we might need to wait for a while before
        the app is actually created or updated.
        """
        while True:
            try:
                await asyncio.sleep(1)
                app_status = serve.status().applications[self.app_name].status
                match app_status:
                    case "RUNNING":
                        self.app_handle = serve.get_app_handle(self.app_name)
                        self._deployment_status = ModelStatus.RUNNING
                        return self._deployment_status
                    case "DEPLOY_FAILED":
                        self.error_msg = (
                            serve.status()
                            .applications[self.app_name]
                            .deployments[self.wrapper_name]
                            .message
                        )
                        self._deployment_status = ModelStatus.DEPLOY_FAILED
                        return self._deployment_status
                    case "UNHEALTHY":
                        self._deployment_status = ModelStatus.UNHEALTHY
                        return self._deployment_status
            except KeyError:
                self._deployment_status = ModelStatus.NONEXISTENT
                return self._deployment_status

    async def get_cached_status(self) -> ModelStatus:
        """
        Return cached model status if it exists, otherwise check the model status.
        This function sets the app handle and checks for the model health.
        """
        if self._deployment_status is not None:
            return self._deployment_status
        else:
            return await self.check_model_status()
