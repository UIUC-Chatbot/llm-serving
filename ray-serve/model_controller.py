import asyncio
import datetime
from logging import getLogger, Logger
import ray
from ray.serve.exceptions import RayServeException
import time

from config_writer import ConfigWriter
from model_context import ModelContext, ModelStatus, ModelType

"""
Architecture:

    The ModelController is a Ray serve app that manages the model pool. It dynamically loads models
    on demand and performs model switching when necessary. It also handles model deletion requests.
    It updates serve apps by modifying a config file and sending it to the Ray dashboard service.

    Each individual model is wrapped in a ModelApp class and deployed as its own Ray Serve app.
    They have their own endpoints. ModelApp is an abstract class, and the actual model
    implementation is specified by the model_type.

    When a ModelApp is inactive but wants to load itself into GPUs, it sends a request to the
    ModelController to load the model. The ModelController might select victim models to evict from
    GPUs and load the requested model.

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
    def __init__(self, config_file_path: str, has_autoscaler: bool) -> None:
        self._config_writer: ConfigWriter = ConfigWriter(config_file_path)
        self._has_autoscaler: bool = has_autoscaler
        self._last_exhaustion_time: float = 0
        self._logger: Logger = getLogger("ray.serve")
        self._model_pool: dict[str, ModelContext] = {}  # Currently registered models
        self._num_gpus: int = int(ray.cluster_resources().get("GPU", 0))
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
        if self._has_autoscaler:
            self._logger.info("ModelController is running with autoscaler enabled.")
        else:
            self._logger.info("ModelController is running without autoscaler.")
        self._logger.info(f"ModelController found {self._num_gpus} GPUs.")
        self._logger.info("ModelController initialized.")

    async def update_num_gpus(self) -> int:
        async with self._lock:
            self._num_gpus = int(ray.cluster_resources().get("GPU", 0))
        self._logger.info(f"Number of GPUs set to {self._num_gpus}.")
        return self._num_gpus

    def count_available_gpus(self) -> int:
        """
        Return the number of available GPUs.
        It might return a negative number if the used GPUs exceed the total number of GPUs, which
        usually happens when some nodes are down.
        """
        used_gpus: int = 0
        for model in self._model_pool.values():
            used_gpus += model.num_active_replicas * model.gpus_per_replica
        return self._num_gpus - used_gpus

    async def _validate_deployment(self, model: ModelContext) -> bool:
        """
        Validate the deployment of the given model. This function should be called each time a model is deployed.
        If this function is called concurrently, only one function instance checks the model
        deployment status, and the rest will wait for them to finish.

        - If the model is unhealthy, remove it from the model pool.
        - If the model is pending for a long time, deactivate it.

        Return true if the model is running, false otherwise.
        """

        def is_model_in_pool(model_context: ModelContext) -> bool:
            # Check whether the model app exists because someone else might have removed it.
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
            model_status: ModelStatus = await model.check_deployment_status()
            self._logger.debug(f"App {model.app_name} is {model_status}.")

            if model_status == ModelStatus.RUNNING:
                model.is_owned = False
                model.is_deployment_success = True
                return True

            elif model_status == ModelStatus.PENDING:
                # The model has been waiting for resources for a long time, deactivate it.
                async with self._lock:
                    self._logger.warning(
                        f"App {model.app_name} has been pending due to resource exhaustion, deactivate it."
                    )
                    self._last_exhaustion_time = time.time()
                    self._deactivate_models([model])

            else:
                async with self._lock:
                    model.is_owned = False
                    model.is_deployment_success = False
                    if not is_model_in_pool(model):
                        return False
                    self._logger.warning(f"App {model.app_name} is {model_status}.")
                    self._config_writer.remove_app(model)
                    self._model_pool.pop(model.model_name)
                    return False

    async def get_or_register_model(
        self,
        model_name: str,
        model_type: ModelType,
        priority: int,
        gpus_per_replica: int = 1,
        hf_key: str | None = None,
        force_load: bool = False,
    ) -> ModelContext | None:
        """
        Return a healthy model_context of the requested model. Create a serve app for the model if
        it's not created yet. Return None if the model deployment is unhealthy.
        """

        async def validate_model(model: ModelContext) -> bool:
            return await asyncio.shield(self._validate_deployment(model))

        if model_name in self._model_pool:
            model: ModelContext = self._model_pool[model_name]
            if not await validate_model(model):  # Model unhealthy
                return None
            if model.num_active_replicas == 0 and force_load:
                # If the model is inactive and force_load is True, try to load it into GPUs.
                async with self._lock:
                    self._activate_model(model)
                if not await validate_model(model):
                    return None
            return model

        # Create a new serve app for the requested model
        async with self._lock:
            # Check again because someone else might have added the model before we woke up.
            if model_name not in self._model_pool:
                model = ModelContext(
                    app_name=f"{model_name.replace('/', '--')}--{self._num_served_models}",
                    model_name=model_name,
                    model_type=model_type,
                    priority=priority,
                    route_prefix=f"/model-{self._num_served_models}",
                    gpus_per_replica=gpus_per_replica,
                )
                model.deployment_status_reset()
                self._num_served_models += 1
                if (
                    self.count_available_gpus() >= gpus_per_replica
                    or force_load
                    or self._has_autoscaler
                ):
                    model.num_active_replicas = 1
                    self._config_writer.add_app(
                        model=model, is_active=True, hf_key=hf_key
                    )
                else:
                    model.num_active_replicas = 0
                    self._config_writer.add_app(
                        model=model, is_active=False, hf_key=hf_key
                    )
                self._model_pool[model_name] = model

        if await validate_model(model):
            return model
        else:
            return None

    async def delete_model_by_model_name(self, model_name: str) -> bool:
        """
        Delete the model app with the given model name.
        """
        async with self._lock:
            if model_name in self._model_pool:
                self._config_writer.remove_app(self._model_pool[model_name])
                self._model_pool.pop(model_name)
                return True
            return False

    async def delete_model_by_app_name(self, app_name: str) -> bool:
        """
        Delete the model app with the given app name.
        """
        async with self._lock:
            for model in self._model_pool.values():
                if model.app_name == app_name:
                    self._config_writer.remove_app(model)
                    self._model_pool.pop(model.model_name)
                    return True
            return False

    async def reset_all(self) -> None:
        """
        Reset LLM services.
        """
        async with self._lock:
            all_models = [model for model in self._model_pool.values()]
            self._config_writer.remove_apps(all_models)
            self._model_pool.clear()
            self._logger.info("LLM service reset.")

    def _activate_model(self, model: ModelContext) -> None:
        """
        It's the caller's responsibility to ensure the availability of GPU resources.
        This function does not verify the availability of any GPU.
        """
        self._config_writer.activate_app(model)
        model.num_active_replicas = 1
        model.activation_status_reset()
        model.deployment_status_reset()
        # Start a background coroutine to check if the model deployment is healthy
        background_task = asyncio.create_task(self._validate_deployment(model))
        asyncio.shield(background_task)

    def _deactivate_models(self, models: list[ModelContext]) -> None:
        self._config_writer.deactivate_apps(models)
        for model in models:
            model.num_active_replicas = 0
            model.deployment_status_reset()
            # Start a background coroutine to check if the model deployment is healthy
            background_task = asyncio.create_task(self._validate_deployment(model))
            asyncio.shield(background_task)

    async def _gather_metrics(
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
    ) -> list[ModelContext] | None:
        """
        The initiator model has requested to load itself into GPU. However, there is no available
        GPU. This function selects a victim model to evict from GPU.

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
            candidates.append(self._gather_metrics(model))

        candidate_reports = await asyncio.gather(*candidates)
        available_candidates: list[tuple[ModelContext, float]] = [
            candidate for candidate in candidate_reports if candidate is not None
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

        # TODO: do we need to wait here for a while? What if the victims don't fulfill the requirement?
        if num_gpus_to_release + available_gpus < required_gpus:
            return None
        else:
            return victims

    async def handle_unavailable_model(self, model_name: str) -> None:
        """
        This function is called by an inactive ModelApp who wants to load itself into GPU.

        It tries its best to unload some models to make room for the requested model, but there is
        no guarantee that the requested model will be loaded into GPU.
        """
        model = self._model_pool[model_name]

        if model.num_active_replicas > 0:
            return  # If the model is already activated, ignore this request.

        async with self._lock:
            if model.num_active_replicas > 0:
                return  # If the model is already activated, ignore this request.

            self._logger.info(f"Trying to load model {model_name} into GPUs.")

            available_gpus = self.count_available_gpus()
            if available_gpus >= model.gpus_per_replica:
                return self._activate_model(model)

            self._logger.info(
                f"Model {model_name} requires {model.gpus_per_replica - available_gpus} more GPUs, which are not available."
            )

            # At this point, there are no resources available for the model.
            # Let's try deploying it anyway and see if the auto-scaler can allocate more resources.
            if (
                self._has_autoscaler
                and not model.activation_failed
                and time.time() - self._last_exhaustion_time > 300
            ):
                self._logger.info(
                    f"Trying to activate model {model_name} even if there are no available GPUs."
                )
                return self._activate_model(model)

            # Auto-scaler doesn't have enough resources, we need to evict some victims from GPUs.
            self._logger.info(
                f"Trying to evict some models to make room for model {model_name}."
            )
            victims: list[ModelContext] | None = await self._select_victim(
                initiator=model,
                required_gpus=model.gpus_per_replica,
                available_gpus=available_gpus,
            )
            if victims is None:
                self._logger.info(
                    f"Resource unavailable for activating model {model_name}."
                )
            else:
                self._deactivate_models(victims)
                self._activate_model(model)

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
                candidates.append(self._gather_metrics(model))
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

    def get_current_config(self) -> dict:
        return self._config_writer.get_current_config()

    def get_model_pool(self) -> dict[str, ModelContext]:
        return self._model_pool
