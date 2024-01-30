from abc import ABC, abstractmethod
from pydantic import BaseModel


class ModelAppInterface(ABC):
    @abstractmethod
    def reconfigure(self, config: dict) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config yaml file. Refer to Ray documentation for more details.
        """
        pass

    @abstractmethod
    def collect_eviction_defense_metrics(self) -> dict:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        pass


class ModelAppArgs(BaseModel):
    model_name: str
    controller: str
