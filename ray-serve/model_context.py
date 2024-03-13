import asyncio
from enum import Enum
from ray import serve
from ray.serve.handle import DeploymentHandle
import time


class ModelStatus(Enum):
    RUNNING = 1  # The model is running and healthy
    PENDING = 2  # The model is waiting for resources
    DEPLOY_FAILED = 3  # The latest model deployment failed
    UNHEALTHY = 4  # The model is currently unhealthy after a successful deployment
    NONEXISTENT = 5  # The model does not exist


class ModelType(Enum):
    EMPTY = 0  # Empty model, for testing purposes
    VLLM_RAW = 1  # Raw VLLM model, created by llm = LLM(model="model_name")
    VLLM_OPENAI = 2  # VLLM OpenAI-Compatible server
    EMBEDDING = 3  # Embedding model


class ModelPath:
    @staticmethod
    def get_import_path(model_type: ModelType) -> str:
        """
        Return a path to the model implementation file.
        """
        if model_type == ModelType.EMPTY:
            return "models.empty:app_builder"

        elif model_type == ModelType.VLLM_RAW:
            return "models.vllm_raw:app_builder"

        elif model_type == ModelType.VLLM_OPENAI:
            return "models.vllm_openai.vllm_openai:app_builder"

        elif model_type == ModelType.EMBEDDING:
            return "models.embedding:app_builder"

        else:
            raise ValueError("Invalid model type.")


class ModelContext:
    def __init__(
        self,
        app_name: str,
        model_name: str,
        model_type: ModelType,
        priority: int,
        route_prefix: str,
        gpus_per_replica: int,
    ) -> None:
        self.app_name: str = app_name
        self.app_handle: DeploymentHandle
        self.error_msg: str = ""  # error message, and possibly stack traces
        self.is_deployment_success: bool | None = None
        self.is_owned: bool = False
        self.model_name: str = model_name
        self.model_type: ModelType = model_type
        self.priority: int = priority  # The higher the number, the higher the priority
        self.route_prefix: str = route_prefix
        self.activation_failed: bool = False
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment
        self.created_time: int = int(time.time())
        self.gpus_per_replica: int = gpus_per_replica
        self.num_active_replicas: int = 0

    def activation_status_reset(self) -> None:
        self.activation_failed = False

    def deployment_status_reset(self) -> None:
        self.is_deployment_success = None

    async def check_deployment_status(self) -> ModelStatus:
        """
        Note that app updates via config file are async, so we might need to wait for a while before
        the app is actually created or updated.
        """
        exhaustion_count: int = 0
        missing_count: int = 0
        while True:
            await asyncio.sleep(1)
            apps = serve.status().applications

            if self.app_name not in apps:
                missing_count += 1
                if missing_count > 5:
                    return ModelStatus.NONEXISTENT
                continue
            missing_count = 0

            app_status = apps[self.app_name].status

            if app_status == "RUNNING":
                self.app_handle = serve.get_app_handle(self.app_name)
                return ModelStatus.RUNNING

            elif app_status == "DEPLOYING":
                """
                Ideally, Ray should provide an API which returns whether the deployment is waiting
                for resources or is just spending a long time in the initialization.
                Unfortunately, they don't, so we have to use a nasty way to check for this.
                """
                try:
                    msg = apps[self.app_name].deployments[self.wrapper_name].message
                    if "Resources required for each replica:" in msg:  # No resources
                        exhaustion_count += 1
                    else:
                        exhaustion_count = 0
                    if exhaustion_count > 90:
                        self.activation_failed = True
                        return ModelStatus.PENDING
                except KeyError:
                    continue

            elif app_status == "DEPLOY_FAILED":
                try:
                    self.error_msg = (
                        apps[self.app_name].deployments[self.wrapper_name].message
                    )
                    return ModelStatus.DEPLOY_FAILED
                except:
                    self.error_msg = "Unknown error."
                    return ModelStatus.DEPLOY_FAILED

            elif app_status == "UNHEALTHY":
                return ModelStatus.UNHEALTHY
