import argparse
import logging
from logging import Logger
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from vllm import LLM, SamplingParams


def get_args():
    parser = argparse.ArgumentParser(description="Ray Serve Model Pool Prototype")
    parser.add_argument(
        "--num-gpus", help="Number of available GPUs", type=int, required=True
    )
    return parser.parse_args()


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=1,
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

    async def __call__(self, prompts: list[str] | str) -> str:
        if type(prompts) == str:
            prompts = [prompts]
        outputs = []
        model_outputs = self.model.generate(prompts, self.sampling_params)
        for output in model_outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            outputs.append(
                f"Model {self.model_name}, Prompt: {prompt!r}, Generated text: {generated_text!r}"
            )
        return outputs


class _ModelInfo:
    def __init__(self, app_name: str) -> None:
        self.app_name: str = app_name
        self.used_gpus: int = 0
        self.last_served_request: int = -1


@serve.deployment
class ModelPool:
    """
    To build llm-serving on top of ray serve instead of modifying serve implementation directly,
    we need to keep track of the number of available GPUs and the number of models served.

    Assuming each model only has one replica (one GPU), and serve autoscaling is disabled, we use an integer to indicate the model's status: 1 indicates the model is active with a deployed serve app; 0 indicates the model is inactive without any deployed serve app.

    There might be some synchronous issues when switching models.
    """

    def __init__(self, num_gpus: int) -> None:
        self._logger: Logger = logging.getLogger("ray.serve")
        self._num_gpus: int = num_gpus
        self._model_pool: dict[str, _ModelInfo] = {}
        self._num_served_models: int = 0
        self._num_served_requests: int = 0

    def _has_available_gpu(self) -> bool:
        """
        Check if there is any available GPU.
        """
        used_gpus: int = 0
        for model_name in self._model_pool.keys():
            used_gpus += self._model_pool[model_name].used_gpus
        return self._num_gpus > used_gpus

    def _get_least_recently_served_model(self) -> str:
        active_models = {k: v for k, v in self._model_pool.items() if v.used_gpus > 0}
        least_recently_served_model = min(
            active_models, key=lambda k: active_models[k].last_served_request
        )
        return least_recently_served_model

    def _switch_model(self, model_in: str, model_out: str = None) -> None:
        """
        Assume both model_in and model_out are already registered in the model pool except when model_out is None. It's the caller's responsibility to ensure the availablity of GPU resources. This function does not verify if model_out is currently utilizing GPU resource,  nor does it check for the availablity of GPUs.
        """
        if model_out is not None:
            serve.delete(self._model_pool[model_out].app_name)
            self._model_pool[model_out].used_gpus = 0
        new_model = VLLM_Model.bind(model_name=model_in)
        serve.run(
            new_model,
            name=self._model_pool[model_in].app_name,
            route_prefix=f"/{self._model_pool[model_in].app_name}",
        )
        self._model_pool[model_in].used_gpus = 1

    async def _call_model(self, model_name: str, prompt: list[str] | str) -> str:
        """
        Send a request to the model with the given name. Load the model if it's not loaded yet.
        """
        if model_name not in self._model_pool:
            model_info = _ModelInfo(f"model-{self._num_served_models}")
            self._num_served_models += 1
            self._model_pool[model_name] = model_info
        self._model_pool[model_name].last_served_request = self._num_served_requests
        self._num_served_requests += 1
        if self._model_pool[model_name].used_gpus == 0:
            if self._has_available_gpu():
                self._switch_model(model_in=model_name, model_out=None)
            else:
                self._switch_model(
                    model_in=model_name,
                    model_out=self._get_least_recently_served_model(),
                )
        model_handle: DeploymentHandle = serve.get_app_handle(
            self._model_pool[model_name].app_name
        )
        return await model_handle.remote(prompt)

    def _delete_model(self, model_name: str) -> str:
        """
        Delete the model with the given name.
        """
        if model_name in self._model_pool:
            if self._model_pool[model_name].used_gpus > 0:
                serve.delete(self._model_pool[model_name].app_name)
            self._model_pool.pop(model_name)
            return f"Model {model_name} deleted."
        else:
            return f"Model {model_name} not found."

    async def __call__(self, request: Request) -> str:
        """
        Parse the input message and call the requested model.
        The input should be in json format, containing the following fields:
            "mode": "call", "delete"
            "model_name": the model name as recognized by VLLM or HuggingFace
            "prompt": the user-provided prompt for the model
        """
        input_msg = await request.json()
        mode: str = input_msg["mode"]
        model_name: str = input_msg["model_name"]

        if mode == "call":
            prompt: list[str] | str = input_msg["prompt"]
            return await self._call_model(model_name, prompt)
        elif mode == "delete":
            return self._delete_model(model_name)
        else:
            return "Invalid mode. Aborting."


if __name__ == "__main__":
    args = get_args()
    ray.init(address="auto")
    main_app_name = "llm-serving"
    pool = ModelPool.bind(args.num_gpus)
    serve.run(pool, name=main_app_name, route_prefix="/llm")
    print(f"{main_app_name} by default is running on http://127.0.0.1:8000/llm")
