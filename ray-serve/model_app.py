from ingress import app
from pydantic import BaseModel
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
import time
from typing import Any
from vllm import LLM, SamplingParams


class UserRequest(BaseModel):
    prompt: str
    load_required: bool = False


@serve.deployment(
    name="ModelApp",
)
@serve.ingress(app)
class ModelApp:
    """
    A thin wrapper class for the model to be served.
    """

    def __init__(self, model_name: str, controller: str) -> None:
        self._controller_app: DeploymentHandle = serve.get_app_handle(controller)
        self._is_active: bool = False
        self._model: LLM | None = None
        self._model_name: str = model_name
        self._last_served_time = time.time()
        self._sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    @app.post("/")
    async def call(self, request: UserRequest) -> str | list[str]:
        if self._is_active:
            self._last_served_time = time.time()
            prompt: list[str] | str = request.prompt
            if type(prompt) == str:
                prompt: list[str] = [prompt]
            outputs = []
            model_outputs = self._model.generate(prompt, self._sampling_params)
            for output in model_outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                outputs.append(
                    f"Model {self._model_name}, Prompt: {prompt!r}, Generated text{generated_text!r}"
                )
            return outputs
        else:
            load_required = request.load_required
            if load_required:
                await self._controller_app.handle_unavailable_model.remote(
                    self._model_name
                )
                return "Service restoring."
            else:
                return "Service temporarily unavailable."

    def reconfigure(self, config: dict[str, Any]) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config yaml file. Refer to Ray documentation for more details.
        """
        self._is_active: bool = config["is_active"]
        if self._is_active:
            self._model = LLM(model=self._model_name)
        else:
            self._model = None

    def collect_eviction_defense_metrics(self) -> dict[str, Any]:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        return {"last_served_time": self._last_served_time}


class ModelAppArgs(BaseModel):
    model_name: str
    controller: str


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.controller)
