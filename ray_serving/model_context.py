import asyncio
from datetime import datetime
from enum import Enum
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import ApplicationDetails, ReplicaDetails, ServeInstanceDetails
import re


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

    def __str__(self) -> str:
        return self.name


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
        num_active_replicas: int,
        gpus_per_replica: int,
        priority: int,
        route_prefix: str,
    ) -> None:
        self.app_name: str = app_name
        self.app_handle: DeploymentHandle
        self.created_time: str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.error_msg: str = (
            "Unknown error"  # deployment error message, and possibly stack traces
        )
        self.gpus_per_replica: int = gpus_per_replica
        self.is_deployment_success: bool | None = None
        self.is_owned: bool = False
        self.model_name: str = model_name
        self.model_type: ModelType = model_type
        self.num_active_replicas: int = num_active_replicas
        self.num_pending_replicas: int = 0
        self.priority: int = priority  # The higher the number, the higher the priority
        self.route_prefix: str = route_prefix
        self.wrapper_name: str = "ModelApp"  # The name of the model deployment

    def reset_deployment_status(self) -> None:
        self.is_deployment_success = None

    def reset_pending_replica_status(self) -> None:
        self.num_pending_replicas = 0

    def _check_deployment_status_once(
        self, dashboard_port: int, check_all: bool
    ) -> ModelStatus | None:
        """
        Check the deployment status of the model once.
        If check_all is True, check all model replicas.
        If check_all is False, return RUNNING status as long as one replica is running.
        """
        try:
            # Use the dashboard to get the application details
            dashboard_address = f"http://localhost:{dashboard_port}"
            serve_details = ServeInstanceDetails(
                **ServeSubmissionClient(dashboard_address).get_serve_details()
            )
            apps: dict[str, ApplicationDetails] = serve_details.applications

            if self.app_name not in apps:
                return ModelStatus.NONEXISTENT

            app_details: ApplicationDetails = apps[self.app_name]
            app_status = app_details.status
            deployment_details = app_details.deployments[self.wrapper_name]
            deployment_msg = deployment_details.message

            if app_status == "RUNNING":  # All model replicas are deployed and running
                self.app_handle = serve.get_app_handle(self.app_name)
                return ModelStatus.RUNNING

            elif app_status == "DEPLOY_FAILED":
                self.error_msg = deployment_msg
                return ModelStatus.DEPLOY_FAILED

            elif app_status == "UNHEALTHY":
                self.error_msg = deployment_msg
                return ModelStatus.UNHEALTHY

            elif app_status == "DEPLOYING":
                if not check_all:
                    replica_list: list[ReplicaDetails] = deployment_details.replicas
                    for replica in replica_list:
                        if replica.state == "RUNNING":
                            # If any replica is running, the app is considered running
                            # The other replicas are still initializing or waiting for resources
                            self.app_handle = serve.get_app_handle(self.app_name)
                            return ModelStatus.RUNNING

                """
                We use the deployment message to determine whether the deployment is pending due to resource constraints and the number of pending replicas.
                Note that if Ray Serve changes the deployment message format, this code will break.
                """
                if "Resources required for each replica:" not in deployment_msg:
                    return None  # Not due to resource constraints, initialization still in progress

                # Use regular expression to find the number of pending replicas
                replica_match = re.search(
                    r"(\d+) replicas that have taken more than",
                    deployment_msg,
                )
                if replica_match:
                    self.num_pending_replicas = int(replica_match.group(1))
                return ModelStatus.PENDING  # No resources available

            else:
                return None

        except:
            return None

    async def check_deployment_status(
        self, dashboard_port: int, check_all: bool
    ) -> ModelStatus:
        """
        Note that app updates via config file are async, so we might need to wait for a while before
        the app is actually created or updated.

        If check_all is True, check all model replicas.
        If check_all is False, return RUNNING status as long as one replica is running.
        """
        count_missing: int = 0
        count_pending: int = 0

        while True:
            await asyncio.sleep(1)

            status = self._check_deployment_status_once(dashboard_port, check_all)
            if status is None:  # The model is either not started or still initializing
                count_missing = 0
                count_pending = 0
                continue

            if status == ModelStatus.NONEXISTENT:
                count_missing += 1
                if count_missing > 10:
                    return ModelStatus.NONEXISTENT
                continue
            count_missing = 0

            if status != ModelStatus.PENDING:
                return status

            count_pending += 1
            if count_pending > 180:  # Pending for too long
                return ModelStatus.PENDING
