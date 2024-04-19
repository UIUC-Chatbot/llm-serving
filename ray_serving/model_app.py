from abc import ABC, abstractmethod
from pydantic import BaseModel

"""
This is an abstract class that forces all model implementations to have some primitive methods used
by our system.

Each model can be in one of the two states: active or inactive.
    - When a model is active, it's loaded onto GPUs and ready to serve requests.
    - When a model is inactive, it's not loaded but the model endpoint is still available. User can
      still send requests to the model endpoint, but the requests will be blocked until the model is
      activated.

Why do we need inactive state? Why not just delete the model app?
Well, if all user requests go through the ModelController and are then forwarded to the model app,
then we indeed don't need the inactive state. But if we want to allow users to send requests
directly to the model app's endpoint, then we need the inactive state, otherwise, users would get
endpoint not found error.

So when do we want users to send requests directly to the model app's endpoint? Possibly when we
have multiple nodes and each node is publicly accessible. In that case, users can send requests
directly to the node that has the model app loaded. This way, we don't need to go through an extra
proxy.
"""


class ModelAppInterface(ABC):
    @abstractmethod
    async def _check_model_availability(self) -> bool:
        """
        This method should be called before serving a request. It checks whether the model is
        active and ready to serve requests. If the model is inactive, it should call the
        ModelController to load the model onto GPU.
        """
        pass

    @abstractmethod
    def reconfigure(self, config: dict) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config
        yaml file or when deployed. Refer to Ray documentation for more details.
        """
        pass

    @abstractmethod
    def collect_eviction_defense_metrics(self) -> dict:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It
        returns metrics that justify the model's continued operation on GPU.
        Note that if this model has multiple replicas, then according to the Ray Serve design, only
        one replica will respond to this call.
        """
        pass


class ModelAppArgs(BaseModel):
    model_name: str
    main: str
    gpus_per_replica: int
