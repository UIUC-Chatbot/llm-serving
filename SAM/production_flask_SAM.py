import time
from typing import BinaryIO, Union
from flask import Flask, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import gc
import io
import requests
from PIL import Image
import cv2
from transformers import pipeline

app = Flask(__name__)

# Load the model once when the application starts
model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=1)

def show_mask(mask, ax, random_color=False):
  if random_color:
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
  else:
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  ax.imshow(mask_image)
  # del mask
  # gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
    start_time = time.monotonic()
    show_mask(mask, ax=ax, random_color=True)
    print(f"⏰ Runtime: {(time.monotonic() - start_time):.2f} seconds")
  plt.axis("off")
  # plt.show()
  plt.savefig('output.png')
  del mask
  del masks
  gc.collect()

@app.route('/run', methods=['GET'])
def run():
  full_run_start_time = time.monotonic()
  model_name: str = request.args.get('model', default='', type=str)
  num_points_per_image: str = request.args.get('num_points_per_image', default=64, type=int)
  print(f"MODEL NAME: {model_name}")
  print(f"num_points_per_image: {num_points_per_image}")
  assert type(num_points_per_image) == int, f"num_points_per_image must be int, found {type(num_points_per_image)}"

  # Load image from URL or raw image binary
  img_url_or_raw_image = request.data
  raw_image = None
  if isinstance(img_url_or_raw_image, str):
    raw_image = Image.open(requests.get(img_url_or_raw_image, stream=True).raw).convert("RGB")
  else:
    raw_image = Image.open(io.BytesIO(img_url_or_raw_image)).convert("RGB")

  # RUN MODEL
  start_time = time.monotonic()
  outputs = model(raw_image, points_per_batch=num_points_per_image)
  print(f"⏰ Model Runtime: {(time.monotonic() - start_time):.2f} seconds")
  masks = outputs["masks"]

  # TODO: use NamedTemporaryFile() to hold output image, for concurrent usage
  show_masks_on_image(raw_image, masks)
  print(f"⏰ Full run() runtime: {(time.monotonic() - full_run_start_time):.2f} seconds")
  return send_file('output.png', mimetype='image/png')

@app.route('/test', methods=['GET'])
def test():
  return "hi kastan"

if __name__ == '__main__':
  app.run(debug=True)
