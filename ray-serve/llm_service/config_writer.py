from logging import getLogger, Logger
import os
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import ServeDeploySchema
import yaml

from model_context import ModelContext, ModelPath


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

    def add_app(self, model: ModelContext, is_active: bool) -> None:
        # import_path is the path to the model implementation file.
        import_path: str = ModelPath.get_import_path(model.model_type)

        """
        VLLM supports distributed inference via Ray, i.e., using multiple GPUs for a single model.
        VLLM utilizes Ray placement group for this feature.
        For models that don't use distributed inference, we must specify num_gpus == 1.
        For models that use distributed inference, we must not specify num_gpus, and use placement
        group to specify the number of GPUs. We use strick_pack strategy to force all bundles to be
        on the same node. GPU communications across multiple nodes seems not well supported.
        """
        if model.gpus_per_replica == 1:
            deployments = [
                {
                    "name": model.wrapper_name,
                    "num_replicas": 1,
                    "ray_actor_options": {"num_cpus": 1, "num_gpus": model.used_gpus},
                    "user_config": {"is_active": is_active},
                },
            ]
        else:  # Distributed inference
            if is_active:
                deployments = [
                    {
                        "name": model.wrapper_name,
                        "num_replicas": 1,
                        "ray_actor_options": {"num_cpus": 1},
                        "placement_group_bundles": [
                            {"CPU": 1, "GPU": 1} for _ in range(model.gpus_per_replica)
                        ],
                        "placement_group_strategy": "STRICT_PACK",
                        "user_config": {"is_active": is_active},
                    },
                ]
            else:
                deployments = [
                    {
                        "name": model.wrapper_name,
                        "num_replicas": 1,
                        "ray_actor_options": {"num_cpus": 1},
                        "placement_group_bundles": [{"CPU": 1}],
                        "placement_group_strategy": "STRICT_PACK",
                        "user_config": {"is_active": is_active},
                    },
                ]

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

    def activate_app(self, model: ModelContext) -> None:
        for app in self._apps:
            if app.get("name") == model.app_name:
                app["deployments"][0]["user_config"]["is_active"] = True
                if model.gpus_per_replica == 1:
                    app["deployments"][0]["ray_actor_options"]["num_gpus"] = 1
                else:
                    app["deployments"][0]["placement_group_bundles"] = [
                        {"CPU": 1, "GPU": 1} for _ in range(model.gpus_per_replica)
                    ]

        self.apply_config()
        self._logger.info(f"App: {model.app_name} activated.")

    def deactivate_apps(self, models: list[ModelContext]) -> None:
        names_model_to_deactivate = [model.app_name for model in models]
        for app in self._apps:
            if app.get("name") in names_model_to_deactivate:
                app["deployments"][0]["user_config"]["is_active"] = False
                if app["args"]["gpus_per_replica"] == 1:
                    app["deployments"][0]["ray_actor_options"]["num_gpus"] = 0
                else:
                    app["deployments"][0]["placement_group_bundles"] = [{"CPU": 1}]

        self.apply_config()
        self._logger.info(f"Apps: {names_model_to_deactivate} deactivated.")

    def get_current_config(self) -> dict:
        return self._config
