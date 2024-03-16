from logging import getLogger, Logger
import os
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import ServeDeploySchema
import yaml

from model_context import ModelContext, ModelPath, ModelType


class ConfigWriter:
    """
    This is a helper class that ModelController uses to update the config file.

    Ray uses config file (yaml) to dynamically update serve apps and deployments.
    ModelController maintains a local copy of the config file, which is updated whenever a model
    is going to be added, removed, activated, or deactivated.
    ModelController sends the config file to the Ray dashboard service, which then updates the
    serve apps and deployments.

    Refer to Ray documentation for more details about the format of the config files.
    """

    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path, "r") as file:
            self._config: dict = yaml.safe_load(file)
        self._apps: list[dict] = self._config["applications"]
        self._logger: Logger = getLogger("ray.serve")

    def apply_config(self) -> None:
        """
        The following code sends a request containing the latest config file to the Ray
        dashboard, which then updates the serve apps and deployments. The call is async; it
        returns immediately without waiting for the dashboard to finish the update.
        """
        try:
            ServeDeploySchema.parse_obj(self._config)
            address: str = os.environ.get(
                "RAY_DASHBOARD_ADDRESS", "http://localhost:8265"
            )
            ServeSubmissionClient(address).deploy_applications(self._config)
        except Exception as e:
            self._logger.error(f"Error applying config: {e}")
            app_set = set()
            deduplicated_apps = []
            for app in self._apps:
                if app.get("name") not in app_set:
                    app_set.add(app["name"])
                    deduplicated_apps.append(app)
            self._config["applications"] = deduplicated_apps
            self._apps = self._config["applications"]
            self._logger.info("Config file deduplicated (if there was duplication).")

    def _configure_empty_model(
        self, model: ModelContext, is_active: bool
    ) -> list[dict]:
        if is_active:
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": {
                        "num_cpus": 1,
                        "num_gpus": model.gpus_per_replica,
                    },
                    "user_config": {"is_active": is_active},
                },
            ]
        else:
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": {
                        "num_cpus": 1,
                        "num_gpus": 0,
                    },
                    "user_config": {"is_active": is_active},
                },
            ]
        return deployments

    def _configure_vllm(
        self, model: ModelContext, is_active: bool, hf_key: str | None = None
    ) -> list[dict]:
        """
        VLLM supports distributed inference via Ray, i.e., using multiple GPUs for a single model.
        VLLM utilizes Ray placement group for this feature.

        For models that don't use distributed inference, we must specify num_gpus == 1 so that the
        model gets a single GPU.

        For models that use distributed inference, we must not specify num_gpus, and instead use
        placement group to allocate the number of GPUs the model requires. Please refer to Ray
        documentation for more details about placement group.
        Simply put, we can specify some bundles of resources in a placement group, and the serve
        deployment will be allocated to the first bundle. Thus, the resources required by the
        deployment itself as specified in ray_actor_options must be available in the first bundle.
        The children Ray actors created by the deployment will be allocated to the placement group
        as well.

        If cross-node GPU communication is not supported, we use STRICT_PACK strategy to force all
        bundles to be on the same node. If we do have cross-node GPU communication, we can use PACK
        strategy to allow the bundles to be allocated to different nodes.
        """
        if model.gpus_per_replica == 1:  # Single GPU
            if is_active:
                num_gpus = 1
            else:
                num_gpus = 0
            if hf_key is not None:
                ray_actor_options = {
                    "num_cpus": 1,
                    "num_gpus": num_gpus,
                    "runtime_env": {"env_vars": {"HF_TOKEN": hf_key}},
                }
            else:
                ray_actor_options = {"num_cpus": 1, "num_gpus": num_gpus}
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": ray_actor_options,
                    "user_config": {"is_active": is_active},
                },
            ]

        else:  # Distributed inference
            placement_group_strategy = "STRICT_PACK"
            if is_active:
                placement_group_bundles = [
                    {"CPU": 1, "GPU": 1} for _ in range(model.gpus_per_replica)
                ]
            else:
                placement_group_bundles = [{"CPU": 1}]
            if hf_key is not None:
                ray_actor_options = {
                    "num_cpus": 1,
                    "runtime_env": {"env_vars": {"HF_TOKEN": hf_key}},
                }
            else:
                ray_actor_options = {"num_cpus": 1}
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": ray_actor_options,
                    "placement_group_bundles": placement_group_bundles,
                    "placement_group_strategy": placement_group_strategy,
                    "user_config": {"is_active": is_active},
                },
            ]
        return deployments

    def _configure_embedding(
        self, model: ModelContext, is_active: bool, hf_key: str | None = None
    ) -> list[dict]:
        if model.gpus_per_replica != 1:
            raise ValueError(
                f"Embedding model {model.app_name} can only use 1 GPU per replica."
            )

        if is_active:
            num_gpus = 1
        else:
            num_gpus = 0
        if hf_key is not None:
            ray_actor_options = {
                "num_cpus": 1,
                "num_gpus": num_gpus,
                "runtime_env": {"env_vars": {"HF_TOKEN": hf_key}},
            }
        else:
            ray_actor_options = {"num_cpus": 1, "num_gpus": num_gpus}

        deployments = [
            {
                "name": model.wrapper_name,
                "num_replicas": 1,
                "ray_actor_options": ray_actor_options,
                "user_config": {"is_active": is_active},
            },
        ]

        return deployments

    def add_app(
        self, model: ModelContext, is_active: bool, hf_key: str | None = None
    ) -> None:
        # import_path is the path to the model implementation file.
        import_path: str = ModelPath.get_import_path(model.model_type)

        # Each model type has their own builder function. Even though this might seem redundant, it
        # makes the code less coupled and more readable.
        deployments: list[dict]
        if model.model_type == ModelType.EMPTY:
            deployments = self._configure_empty_model(model, is_active)

        elif model.model_type == ModelType.VLLM_RAW:
            deployments = self._configure_vllm(model, is_active, hf_key)

        elif model.model_type == ModelType.VLLM_OPENAI:
            deployments = self._configure_vllm(model, is_active, hf_key)

        elif model.model_type == ModelType.EMBEDDING:
            deployments = self._configure_embedding(model, is_active, hf_key)

        else:
            raise ValueError(
                f"Model type {model.model_type} doesn't have a builder function."
            )

        # Add the new app to the config file
        self._apps.append(
            {
                "name": model.app_name,
                "route_prefix": model.route_prefix,
                "import_path": import_path,
                "args": {
                    "model_name": model.model_name,
                    # ModelController is the first app in the config file.
                    "controller": self._config["applications"][0]["name"],
                    "gpus_per_replica": model.gpus_per_replica,
                },
                "deployments": deployments,
            }
        )

        self.apply_config()
        self._logger.info(
            f"App: {model.app_name} added, status: {'active' if is_active else 'inactive'}."
        )

    def remove_app(self, model: ModelContext) -> None:
        self._config["applications"] = [
            app for app in self._apps if app.get("name") != model.app_name
        ]
        self._apps = self._config["applications"]
        self.apply_config()
        self._logger.info(f"App: {model.app_name} removed.")

    def remove_apps(self, models: list[ModelContext]) -> None:
        names_model_to_evict = [model.app_name for model in models]
        self._config["applications"] = [
            app for app in self._apps if app.get("name") not in names_model_to_evict
        ]
        self._apps = self._config["applications"]
        self.apply_config()
        self._logger.info(f"Apps: {names_model_to_evict} removed.")

    def _toggle_empty_model(
        self, model: ModelContext, app: dict, should_be_active: bool
    ) -> None:
        if should_be_active:
            app["deployments"][0]["user_config"]["is_active"] = True
            app["deployments"][0]["ray_actor_options"][
                "num_gpus"
            ] = model.gpus_per_replica

        else:
            app["deployments"][0]["user_config"]["is_active"] = False
            app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0

    def _toggle_vllm(
        self, model: ModelContext, app: dict, should_be_active: bool
    ) -> None:
        if should_be_active:
            app["deployments"][0]["user_config"]["is_active"] = True
            if model.gpus_per_replica == 1:
                app["deployments"][0]["ray_actor_options"]["num_gpus"] = 1
            else:
                app["deployments"][0]["placement_group_bundles"] = [
                    {"CPU": 1, "GPU": 1} for _ in range(model.gpus_per_replica)
                ]
        else:
            app["deployments"][0]["user_config"]["is_active"] = False
            if model.gpus_per_replica == 1:
                app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0
            else:
                app["deployments"][0]["placement_group_bundles"] = [{"CPU": 1}]

    def _toggle_embedding(
        self, model: ModelContext, app: dict, should_be_active: bool
    ) -> None:
        if should_be_active:
            app["deployments"][0]["user_config"]["is_active"] = True
            app["deployments"][0]["ray_actor_options"][
                "num_gpus"
            ] = model.gpus_per_replica

        else:
            app["deployments"][0]["user_config"]["is_active"] = False
            app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0

    def activate_app(self, model: ModelContext) -> None:
        # Each model type has their own toggle function. Even though this might seem redundant, it
        # makes the code less coupled and more readable.
        for app in self._apps:
            if app.get("name") != model.app_name:
                continue

            if model.model_type == ModelType.EMPTY:
                self._toggle_empty_model(model, app, True)

            elif model.model_type == ModelType.VLLM_RAW:
                self._toggle_vllm(model, app, True)

            elif model.model_type == ModelType.VLLM_OPENAI:
                self._toggle_vllm(model, app, True)

            elif model.model_type == ModelType.EMBEDDING:
                self._toggle_embedding(model, app, True)

            else:
                raise ValueError(
                    f"Model type {model.model_type} doesn't have a toggle function."
                )
            break

        self.apply_config()
        self._logger.info(f"App: {model.app_name} activated.")

    def deactivate_apps(self, models: list[ModelContext]) -> None:
        # Each model type has their own toggle function. Even though this might seem redundant, it
        # makes the code less coupled and more readable.
        models_dict = {model.app_name: model for model in models}
        for app in self._apps:
            app_name = app.get("name", None)
            if app.get("name") not in models_dict:
                continue
            model = models_dict[app_name]

            if model.model_type == ModelType.EMPTY:
                self._toggle_empty_model(model, app, False)

            elif model.model_type == ModelType.VLLM_RAW:
                self._toggle_vllm(model, app, False)

            elif model.model_type == ModelType.VLLM_OPENAI:
                self._toggle_vllm(model, app, False)

            elif model.model_type == ModelType.EMBEDDING:
                self._toggle_embedding(model, app, False)

            else:
                raise ValueError(
                    f"Model type {model.model_type} doesn't have a toggle function."
                )

        self.apply_config()
        self._logger.info(f"Apps: {list(models_dict.keys())} deactivated.")

    def get_current_config(self) -> dict:
        return self._config
