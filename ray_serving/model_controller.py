import asyncio
import datetime
from logging import getLogger, Logger
import ray
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.exceptions import RayServeException
from ray.serve.schema import ApplicationDetails, ServeInstanceDetails
import time

from config_writer import ConfigWriter
from model_context import ModelContext, ModelStatus, ModelType

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.
    It updates serve apps by modifying a config file and sending it to the Ray dashboard service.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    Refer to model_app.py for more details. ModelApp is an abstract class, and the actual model
    implementation is specified by the model_type.

    Each deployed model app is associated with a ModelContext object.

    When a higher-priority model is requested but there are no available resources, ModelController
    evicts lower-priority models from GPUs to make room for the requested model.

    A model daemon is deployed on the head node. It periodically checks the health of the model pool
    and the availability of GPUs. If a ModelApp becomes unhealthy, probably due to worker node it is
    running on being down, Ray will try to redeploy the ModelApp on other worker nodes. If, however,
    there is no available resources, then we need to remove those unhealthy ModelApps.
"""

"""
Mental Model:

    The ModelController is a centralized server that manages the model pool. It takes requests from
    users and executes them one at a time. User requests are essentially cooperative coroutines that
    are running in an event loop. Therefore, only one active coroutine is running at a time and it
    yields control to the event loop when it's "await"ing for something.

    Refer to https://docs.ray.io/en/latest/serve/architecture.html#lifetime-of-a-request for more
    details.

    When passing objects between Serve apps, we are actually passing copies of them. Under the hood,
    DeploymentResponse corresponds to a Ray ObjectRef, which is a reference to an immutable
    object in the object store. However, the response we obtain by "await"ing the remote call is a
    copy of the original object in the object store. Therefore, we can't modify the original object
    by modifying the response object.
"""


class ModelController:
    def __init__(
        self, autoscaler_enabled: bool, config_file_path: str, dashboard_port: int
    ) -> None:
        self._autoscaler_enabled: bool = autoscaler_enabled
        self._autoscaler_last_failed_gpus: int = 0
        self._autoscaler_last_failed_time: float = 0
        self._config_writer: ConfigWriter = ConfigWriter(
            config_file_path, dashboard_port
        )
        self._dashboard_port: int = dashboard_port
        self._logger: Logger = getLogger("ray.serve")
        self._model_name_map: dict[str, str] = {}  # app_name -> model_name
        self._model_pool: dict[str, ModelContext] = {}  # Currently registered models
        self._num_gpus_total: int = int(ray.cluster_resources().get("GPU", 0))
        self._num_served_models: int = 0

        """
        Modifications of the model pool should be atomic.
        If some coroutines need to modify the model pool and execute await statements during the
        modification process (yield control to the event loop), they must acquire the lock.
        Read operations don't need to acquire the lock.
        """
        self._lock: asyncio.Lock = asyncio.Lock()

        """
        Apply the initial config, just in case the ModelController is restarted during service for
        some reason, it's better to restart the whole LLM service, since the model_pool is reset to
        empty and all past information is lost.
        (assuming the provided config file only has the default configs)
        """
        self._config_writer.apply_config()
        if self._autoscaler_enabled:
            self._logger.info("ModelController is running with autoscaler enabled.")
        else:
            self._logger.info("ModelController is running without autoscaler.")
        self._logger.info(f"ModelController found {self._num_gpus_total} GPUs.")
        self._logger.info("ModelController initialized.")

    async def update_num_gpus_total(self) -> int:
        async with self._lock:
            self._num_gpus_total = int(ray.cluster_resources().get("GPU", 0))
        self._logger.info(f"Number of GPUs set to {self._num_gpus_total}.")
        return self._num_gpus_total

    def count_available_gpus(self) -> int:
        """
        Return the number of available GPUs.
        This function assumes the caller has acquired the lock.
        It might return a negative number if the used GPUs exceed the total number of GPUs, which
        usually happens when some nodes are down.
        """
        num_gpus_used: int = 0
        for model in self._model_pool.values():
            num_gpus_used += model.num_active_replicas * model.gpus_per_replica
        return self._num_gpus_total - num_gpus_used

    def _is_app_in_pool(self, model_context: ModelContext) -> bool:
        """
        Check whether the model_context is in the model pool.
        """
        model_name: str = model_context.model_name
        if model_name not in self._model_pool:
            return False
        """
        In rare cases, the app corresponding to the current model context has been removed, but
        a new app with the same model name has been created, therefore the model name can be
        found in the model pool, but the app name is different.
        """
        if model_context.app_name != self._model_pool[model_name].app_name:
            return False
        return True

    async def _clear_pending_replicas(self, model: ModelContext) -> None:
        self._logger.debug(
            f"Start clearing the model {model.model_name}'s pending replicas."
        )
        model_status = await model.check_deployment_status(
            self._dashboard_port, check_all=True
        )
        self._logger.debug(f"App {model.app_name} is {model_status}.")

        if model_status == ModelStatus.PENDING:
            # Some replicas have been waiting for resources for too long, downscale the model.
            async with self._lock:
                self._logger.warning(
                    f"App {model.app_name} has {model.num_pending_replicas} replicas that have been pending due to resource exhaustion, downscaling the model."
                )
                self._autoscaler_last_failed_gpus = model.gpus_per_replica
                self._autoscaler_last_failed_time = time.time()
                # Downscale the model
                new_num_active_replicas = (
                    model.num_active_replicas - model.num_pending_replicas
                )
                self._set_num_replicas_for_one_model(model, new_num_active_replicas)

    async def _validate_deployment(self, model: ModelContext) -> bool:
        """
        Validate the deployment of the given model. This function should be called each time a model is deployed.
        If this function is called concurrently, only one function instance checks the model
        deployment status, and the rest will wait for them to finish.

        - If the model is unhealthy, remove it from the model pool.
        - If the model's replicas are pending for a long time, downscale it.

        Return true if the model is running, false otherwise.
        """
        if model.is_deployment_success is not None:
            # The model has been checked before, return the result.
            return model.is_deployment_success

        while model.is_owned:  # Someone else has started checking the model.
            self._logger.debug(
                f"Waiting for the model {model.model_name} to be checked by another coroutine."
            )
            await asyncio.sleep(1)

        if model.is_deployment_success is not None:
            # The model has been checked by someone else, return the result.
            return model.is_deployment_success

        # Only one function instance (owner) checks the model deployment status.
        model.is_owned = True
        self._logger.debug(
            f"Start checking the model {model.model_name} deployment status."
        )
        while True:
            model_status = await model.check_deployment_status(
                self._dashboard_port, check_all=False
            )
            self._logger.debug(f"App {model.app_name} is {model_status}.")

            if model_status == ModelStatus.RUNNING:
                model.is_deployment_success = True
                model.is_owned = False
                # Start a background coroutine to clear any pending replicas
                background_task = asyncio.create_task(
                    self._clear_pending_replicas(model)
                )
                asyncio.shield(background_task)
                return True

            elif model_status == ModelStatus.PENDING:
                # The model been waiting for resources for too long, downscale the model.
                async with self._lock:
                    self._logger.warning(
                        f"App {model.app_name} has {model.num_pending_replicas} replicas that have been pending due to resource exhaustion, downscaling the model."
                    )
                    self._autoscaler_last_failed_gpus = model.gpus_per_replica
                    self._autoscaler_last_failed_time = time.time()

                    # Downscale the model
                    new_num_active_replicas = (
                        model.num_active_replicas - model.num_pending_replicas
                    )
                    self._set_num_replicas_for_one_model(model, new_num_active_replicas)
                    # The model has just been updated, continue checking its deployment status.

            else:  # Model deployment failed
                async with self._lock:
                    model.is_deployment_success = False
                    model.is_owned = False
                    if not self._is_app_in_pool(model):
                        return False  # Someone else has removed the model from the pool
                    self._logger.warning(f"App {model.app_name} is {model_status}.")
                    self._config_writer.remove_app(model)
                    self._model_name_map.pop(model.app_name)
                    self._model_pool.pop(model.model_name)
                    return False

    async def get_or_register_model(
        self,
        model_name: str,
        model_type: ModelType,
        num_replicas: int,
        gpus_per_replica: int,
        priority: int,
        hf_key: str | None = None,
    ) -> ModelContext | str:
        """
        Return a model_context of the requested model. Create a serve app for the model if it's not
        created yet.
        Return an error message if the model deployment fails.

        We do our best to deploy num_replicas replicas of the model, but there is no guarantee that
        all replicas will be deployed successfully, possibly due to resource constraint.
        """

        async def is_model_healthy(model: ModelContext) -> bool:
            # Check the model deployment status
            # asyncio.shield is used to prevent the coroutine from being cancelled.
            return await asyncio.shield(self._validate_deployment(model))

        if model_name in self._model_pool:
            model: ModelContext = self._model_pool[model_name]
            if await is_model_healthy(model):
                return model
            else:
                return model.error_msg

        # Create a new serve app for the requested model
        async with self._lock:
            # Check again because someone else might have added the model before we woke up.
            if model_name not in self._model_pool:
                app_name = f"{model_name.replace('/', '--')}--{self._num_served_models}"
                model = ModelContext(
                    app_name=app_name,
                    model_name=model_name,
                    model_type=model_type,
                    num_active_replicas=num_replicas,
                    gpus_per_replica=gpus_per_replica,
                    priority=priority,
                    route_prefix=f"/model-{self._num_served_models}",
                )
                model.reset_deployment_status()
                model.reset_pending_replica_status()
                self._num_served_models += 1

                # Adjust the number of replicas if the autoscaler is disabled
                if not self._autoscaler_enabled:
                    max_num_replicas = self.count_available_gpus() // gpus_per_replica
                    if model.num_active_replicas > max_num_replicas:
                        model.num_active_replicas = max_num_replicas

                # add_app first, then update the pool. Just in case add_app throws an exception.
                self._config_writer.add_app(model=model, hf_key=hf_key)
                self._model_name_map[app_name] = model_name
                self._model_pool[model_name] = model

        if await is_model_healthy(model):
            return model
        else:
            return model.error_msg

    async def delete_model_by_model_name(self, model_name: str) -> bool:
        """
        Delete the model app with the given model name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                model_context: ModelContext = self._model_pool[model_name]
                self._config_writer.remove_app(model_context)
                self._model_name_map.pop(model_context.app_name)
                self._model_pool.pop(model_name)
                return True
            return False

    async def delete_model_by_app_name(self, app_name: str) -> bool:
        """
        Delete the model app with the given app name.
        """
        async with self._lock:
            if app_name in self._model_name_map:
                model_name: str = self._model_name_map[app_name]
                model_context: ModelContext = self._model_pool[model_name]
                self._config_writer.remove_app(model_context)
                self._model_name_map.pop(app_name)
                self._model_pool.pop(model_name)
                return True
            return False

    async def reset_all(self) -> None:
        """
        Reset LLM services.
        """
        async with self._lock:
            all_models = [model for model in self._model_pool.values()]
            self._config_writer.remove_apps(all_models)
            self._model_name_map.clear()
            self._model_pool.clear()
            self._logger.info("LLM service reset.")

    def _set_num_replicas_for_one_model(
        self, model: ModelContext, num_replicas: int
    ) -> None:
        """
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify the availability of any GPU.
        """
        model.num_active_replicas = num_replicas
        self._config_writer.update_apps([model])
        model.reset_deployment_status()
        model.reset_pending_replica_status()
        # Start a background coroutine to check if the model deployment is healthy
        background_task = asyncio.create_task(self._validate_deployment(model))
        asyncio.shield(background_task)

    def _set_num_replicas_for_model_list(
        self, models: list[ModelContext], num_replica_list: list[int]
    ) -> None:
        """
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify the availability of any GPU.
        """
        for idx, model in enumerate(models):
            model.num_active_replicas = num_replica_list[idx]

        self._config_writer.update_apps(models)

        for model in models:
            model.reset_deployment_status()
            model.reset_pending_replica_status()
            # Start a background coroutine to check if the model deployment is healthy
            background_task = asyncio.create_task(self._validate_deployment(model))
            asyncio.shield(background_task)

    async def _gather_defense_metrics(
        self, model: ModelContext
    ) -> tuple[ModelContext, float] | None:
        try:
            """
            The caller of this function owns the lock, so we don't want this function to block
            for too long. We set a timeout and ignore the model if it doesn't respond in time.
            """
            metrics = await asyncio.wait_for(
                model.app_handle.collect_eviction_defense_metrics.remote(),
                timeout=10,
            )
            return model, metrics["last_served_time"]
        except (
            AttributeError,
            RayServeException,
            TimeoutError,
            asyncio.TimeoutError,
        ):
            return None

    async def _select_victim(
        self, initiator: ModelContext, required_gpus: int, available_gpus: int
    ) -> tuple[list[ModelContext], list[int]] | None:
        """
        The initiator model has requested to use required_gpus GPUs. This function selects victim
        models to evict from GPUs.

        This function calls each model's collect_eviction_defense_metrics method to get the latest
        service information.
        """

        candidates = []
        for model in self._model_pool.values():
            if model.app_name == initiator.app_name:
                continue
            if model.num_active_replicas == 0:
                continue
            if model.priority > initiator.priority:
                continue
            candidates.append(self._gather_defense_metrics(model))

        candidate_reports = await asyncio.gather(*candidates)
        available_candidates: list[tuple[ModelContext, float]] = [
            candidate_report
            for candidate_report in candidate_reports
            if candidate_report is not None
        ]

        # Remove the least recently used model
        available_candidates.sort(key=lambda x: x[1])
        num_gpus_to_release = 0
        victims: list[ModelContext] = []
        for candidate in available_candidates:
            victims.append(candidate[0])
            num_gpus_to_release += (
                candidate[0].num_active_replicas * candidate[0].gpus_per_replica
            )
            if num_gpus_to_release + available_gpus >= required_gpus:
                break

        # TODO: do we need to wait here for a while?
        if num_gpus_to_release + available_gpus < model.gpus_per_replica:
            # We can't allocate enough resource even for one replica, give up.
            return None
        else:
            return victims, [0 for _ in range(len(victims))]

    async def load_model(self, model_name: str, num_replicas: int) -> str:
        """
        Load the model app with the given model name into GPUs.

        We do our best to deploy num_replicas replicas of the model, but there is no guarantee that
        all replicas will be deployed successfully, e.g., higher-priority models might have been
        loaded into GPUs.
        """
        if num_replicas < 0:
            return "Number of replicas must be non-negative."

        if model_name not in self._model_pool:
            return f"Model {model_name} is not registered."

        model: ModelContext = self._model_pool[model_name]
        if model.num_active_replicas == num_replicas:  # Ignore the request
            return f"Model {model_name} already has {num_replicas} replicas."

        async with self._lock:
            if model.num_active_replicas == num_replicas:  # Ignore the request
                return f"Model {model_name} active replicas set to {num_replicas}."

            self._logger.info(
                f"Trying to load {num_replicas} replicas of {model_name} into GPUs."
            )

            available_gpus = self.count_available_gpus()
            available_gpus += model.num_active_replicas * model.gpus_per_replica
            required_gpus = num_replicas * model.gpus_per_replica
            if available_gpus >= required_gpus:
                self._set_num_replicas_for_one_model(model, num_replicas)
                return f"Model {model_name} active replicas set to {num_replicas}."

            # At this point, there are no resources available for the model.
            self._logger.info(
                f"Model {model_name} requires {required_gpus - available_gpus} more GPUs, which are not available."
            )

            if self._autoscaler_enabled:
                """
                Let's try deploying it anyway and see if the autoscaler helps.
                Skip autoscaling if recent attempt failed and our model requires more GPUs for each
                replica than the failed attempt.
                """
                if (
                    model.gpus_per_replica < self._autoscaler_last_failed_gpus
                    or time.time() - self._autoscaler_last_failed_time > 300
                ):
                    self._logger.info(
                        f"Set model {model_name} active replicas to {num_replicas} even if there are no available GPUs."
                    )
                    self._set_num_replicas_for_one_model(model, num_replicas)
                    return f"Model {model_name} active replicas set to {num_replicas}."

            # We don't have enough resources. Let's pick some victim models to evict from GPUs.
            self._logger.info(
                f"Trying to evict some models to make room for model {model_name}."
            )
            victims: tuple[list[ModelContext], list[int]] | None = (
                await self._select_victim(
                    initiator=model,
                    required_gpus=model.gpus_per_replica,
                    available_gpus=available_gpus,
                )
            )
            if victims is None:
                self._logger.info(
                    f"No resources for loading more active replica of model {model_name}."
                )
                return f"No resources for loading more active replica of model {model_name}."
            else:
                self._logger.info(
                    f"Victims selected: {[victim.model_name for victim in victims[0]]}."
                )
                self._set_num_replicas_for_model_list(victims[0], victims[1])
                self._set_num_replicas_for_one_model(model, num_replicas)
                return f"Model {model_name} active replicas set to {num_replicas}."

    async def clean_unpopular_models(self) -> None:
        candidates = []
        unpopular_models: list[ModelContext] = []

        async with self._lock:
            for model in self._model_pool.values():
                if model.priority > 1:
                    now = datetime.datetime.now().time()
                    morning = datetime.time(9, 0, 0)
                    night = datetime.time(23, 0, 0)
                    if now > morning and now < night:
                        continue
                candidates.append(self._gather_defense_metrics(model))
            candidate_reports = await asyncio.gather(*candidates)
            available_candidates: list[tuple[ModelContext, float]] = [
                candidate for candidate in candidate_reports if candidate is not None
            ]

            now = time.time()
            for candidate in available_candidates:
                # The model has not been used for a long time
                if now - candidate[1] > 600:
                    unpopular_models.append(candidate[0])

        # Release the lock and deactivate the unpopular models
        for model in unpopular_models:
            self._logger.info(
                f"Remove {model.model_name} because it has not been used for a long time."
            )
            await self.delete_model_by_model_name(model.model_name)

    def _update_model_num_replicas(
        self, app_name: str, app_details: ApplicationDetails
    ) -> None:
        """
        Check the number of replicas of the given model app. Update the model context if the number
        of replicas has changed.
        """
        model_name: str | None = self._model_name_map.get(app_name)
        if model_name is None:
            self._logger.warning(
                f"App {app_name} does not correspond to any model in the model pool."
            )
            return

        try:
            num_replicas: int = len(app_details.deployments["ModelApp"].replicas)
        except KeyError:  # The Model App hasn't been stabilized yet, ignore it
            return

        model_context: ModelContext = self._model_pool[model_name]
        if model_context.num_active_replicas == 0:  # Model is inactive
            if num_replicas > 1:
                self._logger.warning(
                    f"Model {model_name} has {num_replicas} replicas but is not active."
                )
        else:  # Model is active
            if model_context.num_active_replicas != num_replicas:
                self._logger.info(
                    f"Model {model_name} now has {num_replicas} replicas, previously {model_context.num_active_replicas} replicas."
                )
        model_context.num_active_replicas = num_replicas

    async def update_model_pool(self) -> None:
        """
        Update the number of replicas of each model, since these numbers might change if we enable
        replica autoscaler.
        If replica autoscaler is disabled, this function essentially does nothing, because Ray does
        not automatically scale the number of replicas.
        """
        # Use the dashboard to get the application details
        dashboard_address = f"http://localhost:{self._dashboard_port}"
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient(dashboard_address).get_serve_details()
        )
        apps: dict[str, ApplicationDetails] = serve_details.applications

        async with self._lock:
            for app_name, app_details in apps.items():
                if app_name == "llm-serving" or app_name == "llm-daemon":
                    continue  # Ignore the system apps
                self._update_model_num_replicas(app_name, app_details)

    def get_current_config(self) -> dict:
        return self._config_writer.get_current_config()

    def get_model_pool(self) -> dict[str, ModelContext]:
        return self._model_pool
