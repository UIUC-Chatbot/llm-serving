from typing import List, Union
import torch
from transformers import pipeline

def run_automodel(prompt: Union[str,List[str]], model: str, task: Union[str, None] = None, **pipeline_kwargs):
  """
  Args:
    task (`str` or `List`, *optional*):
        Most popular models work without specifying this. A string or list of strings of tasks models 
        were designed for, such as: "fill-mask" or "automatic-speech-recognition"
  """
  try:
    pipe = pipeline(task=task, model=model, device_map='auto')
    return pipe(prompt, **pipeline_kwargs)
    
  except RuntimeError as e:
    if "out of memory" in str(e):
      print(f"Cuda OOM error: {e}")
      return f"Cuda OOM error: {e}"
    else:
      raise e


def run_custom_model(prompt: Union[str,List[str]], model: str, task: Union[str, None] = None, **pipeline_kwargs):
  """
  Args:
    task (`str` or `List`, *optional*):
        Most popular models work without specifying this. A string or list of strings of tasks models 
        were designed for, such as: "fill-mask" or "automatic-speech-recognition"
  """
  try:

    pipe = pipeline(task=task, model=model, device_map='auto')
    return pipe(prompt, **pipeline_kwargs)
    
  except RuntimeError as e:
    if "out of memory" in str(e):
      print(f"Cuda OOM error: {e}")
      return f"Cuda OOM error: {e}"
    else:
      raise e

if __name__ == "__main__":
  prompt = "How many helicopters can a human eat in one sitting?"
  model = "HuggingFaceH4/zephyr-7b-alpha"
  task_filter = "text-generation"
  pipeline_kwargs = {'max_new_tokens': 256, 'do_sample': True, 'temperature': 0.7, 'top_k': 50, 'top_p': 0.95}
  print(run_automodel(prompt, model, task_filter, **pipeline_kwargs))
