import logging
from logging import Logger
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from vllm import LLM, SamplingParams


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 1},
)
class VLLM_Model:
    # A wrapper class for VLLM models so that each model deployment can be scaled independently.
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.model_name: str = model_name
        self.model: LLM = LLM(model=self.model_name)
        self.sampling_params: SamplingParams = SamplingParams(
            temperature=temperature, top_p=top_p
        )

    async def __call__(self, request: Request) -> str:
        input_msg = await request.json()
        prompts: list[str] | str = input_msg["prompts"]
        if type(prompts) == str:
            prompts = [prompts]
        outputs = []
        model_outputs = self.model.generate(prompts, self.sampling_params)
        for output in model_outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            outputs.append(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return outputs


@serve.deployment
class ModelPool:
    def __init__(self) -> None:
        self._logger: Logger = logging.getLogger("ray.serve")
        self._model_routes: dict[str : tuple(str, str)] = {}
        self._num_served_models: int = 0

    def _get_or_load_model(self, model_name: str) -> DeploymentHandle:
        """
        Get the route prefix of the requested model. Load it if not already loaded.
        """
        if model_name not in self._model_routes:
            new_model = VLLM_Model.bind(model_name=model_name)
            app_name = f"model-{self._num_served_models}"
            prefix = f"/{app_name}"
            serve.run(new_model, name=app_name, route_prefix=prefix)
            self._num_served_models += 1
            self._model_routes[model_name] = (app_name, prefix)
        return self._model_routes[model_name][1]

    def _delete_model(self, model_name: str) -> str:
        """
        Delete the model with the given name.
        """
        if model_name in self._model_routes:
            serve.delete(self._model_routes[model_name][0])
            self._model_routes.pop(model_name)
            return f"Model {model_name} deleted."
        else:
            return f"Model {model_name} not found."

    async def __call__(self, request: Request) -> str:
        """
        Parse the input message and call the requested model.
        """
        input_msg = await request.json()
        mode: str = input_msg["mode"]
        model_name: str = input_msg["model_name"]

        if mode == "get":
            return self._get_or_load_model(model_name)
        elif mode == "delete":
            return self._delete_model(model_name)
        else:
            return "Invalid mode. Aborting."


if __name__ == "__main__":
    ray.init(address="auto")
    main_app_name = "llm-serving"
    pool = ModelPool.bind()
    serve.run(pool, name=main_app_name, route_prefix="/llm")
    print(f"{main_app_name} by default is running on http://127.0.0.1:8000/llm")
