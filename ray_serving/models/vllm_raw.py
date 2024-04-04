import fastapi
from model_app import ModelAppInterface, ModelAppArgs
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


app = fastapi.FastAPI()


@serve.deployment(name="ModelApp")
@serve.ingress(app)
class ModelApp(ModelAppInterface):
    """
    A thin wrapper class for the model to be served.
    """

    def __init__(self, model_name: str, main: str, gpus_per_replica: int) -> None:
        self._main: DeploymentHandle = serve.get_app_handle(main)
        self._is_active: bool = False
        self._model: LLM | None = None
        self._model_name: str = model_name
        self._gpus_per_replica: int = gpus_per_replica
        self._last_served_time: float = time.time()
        self._sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    async def _check_model_availability(self) -> bool:
        if self._is_active:
            return True
        await self._main.handle_unavailable_model.remote(self._model_name)
        return False

    @app.post("/")
    async def call(self, request: UserRequest) -> str | list[str]:
        if not await self._check_model_availability():
            return "Model is not active"
        self._last_served_time = time.time()
        input_data: list[str] = [request.prompt]
        outputs = []
        model_outputs = self._model.generate(input_data, self._sampling_params)  # type: ignore
        for output in model_outputs:
            input_prompt = output.prompt
            generated_text = output.outputs[0].text
            outputs.append(
                f"Model {self._model_name}, Prompt: {input_prompt!r}, Generated te{generated_text!r}"
            )
        return outputs

    def reconfigure(self, config: dict[str, Any]) -> None:
        """
        This method is called when the model is being reconfigured via "user_config" in the config yaml file. Refer to Ray documentation for more details.
        """
        self._is_active: bool = config["is_active"]
        if self._is_active:
            self._model = LLM(
                model=self._model_name,
                tensor_parallel_size=self._gpus_per_replica,
                trust_remote_code=True,
                device="cuda",
            )
        else:
            self._model = None

    def collect_eviction_defense_metrics(self) -> dict[str, Any]:
        """
        This method is called when the ModelController is trying to evict a model from GPU. It returns metrics that justify the model's continued operation on GPU.
        """
        return {"last_served_time": self._last_served_time}


def app_builder(args: ModelAppArgs) -> Application:
    return ModelApp.bind(args.model_name, args.main, args.gpus_per_replica)
