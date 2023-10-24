from typing import DefaultDict, List, Union
from huggingface_hub import HfApi, list_models
from huggingface_hub.hf_api import ModelInfo

def list_automodels(task_filter: Union[str,List[str]], hf_token: str = '' ):
  '''These can be run with just the model name and a prompt, no other inputs required.
  All models & tasks publiclly on the Hub (as of October, 2023)
  Total models on Hub = 356,609
  Total models with pipeline tags = 190,650 (53.5%)
  {
    'text-classification': 33353,
    'reinforcement-learning': 29993,
    'text-generation': 26479,
    'text2text-generation': 18856,
    'automatic-speech-recognition': 11458,
    'token-classification': 11213,
    'text-to-image': 10622,
    'fill-mask': 8858,
    'question-answering': 7278,
    'image-classification': 6008,
    'feature-extraction': 5706,
    'audio-to-audio': 3592,
    'translation': 2896,
    'conversational': 2498,
    'sentence-similarity': 2413,
    'text-to-speech': 1651,
    'summarization': 1337,
    'audio-classification': 1320,
    'object-detection': 897,
    'unconditional-image-generation': 839,
    'multiple-choice': 687,
    'text-to-audio': 385,
    'video-classification': 329,
    'image-segmentation': 292,
    'image-to-text': 283,
    'tabular-classification': 252,
    'image-to-image': 203,
    'zero-shot-image-classification': 202,
    'zero-shot-classification': 180,
    'tabular-regression': 167,
    'visual-question-answering': 82,
    'table-question-answering': 77,
    'depth-estimation': 74,
    'document-question-answering': 70,
    'text-to-video': 49,
    'voice-activity-detection': 17,
    'other': 12,
    'graph-ml': 12,
    'robotics': 9,
    'time-series-forecasting': 1
  }
  '''
  hf_api = HfApi(
    endpoint="https://huggingface.co", 
    token=hf_token, # no token necessary, optionally let the user pass their own token to list private/gated models
  )


  for m in hf_api.list_models():
    if m.pipeline_tag: 

      if task_filter and m.task in task_filter:
        yield m.model_name 
      
      
