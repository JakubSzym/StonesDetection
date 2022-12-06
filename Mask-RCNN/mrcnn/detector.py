#!/usr/bin/env python3

from argparse import ArgumentParser
from stones import StonesConfig

import os
import cv2
import model as modellib
import numpy as np
import time

DEBUG = True

class InferenceConfig(StonesConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

def random_colors(N):
  np.random.seed(1)
  colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
  return colors

def apply_mask(image, mask, color, alpha=0.5):
  for n, c in enumerate(color):
    image[:,:,n] = np.where(mask == 1, image[:,:,n] * (1 - alpha) + alpha * c, image[:,:,n])
  return image

def apply_instances(image, boxes, masks, ids, names, scores):
  n_instances = boxes.shape[0]
  colors = random_colors(n_instances)
  if not n_instances:
    print("NO INSTANCES TO DISPLAY")
  else:
    assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
  for i, color in enumerate(colors):
    if not np.any(boxes[i]):
      continue
    y1, x1, y2, x2 = boxes[i]
    label = names[ids[i]]
    score = scores[i] if scores is not None else None
    caption = '{} {:.2f}'.format(label, score) if score else label
    mask = masks[:, :, i]
    image = apply_mask(image, mask, color)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
  return image

def make_video(out_video, images):
  example_frame = images[0]
  height, width = example_frame.shape[:2]
  video_writer = cv2.VideoWriter(out_video, 0, 25, (width, height))
  for image in images:
    video_writer.write(image)
  video_writer.release()

ROOT_DIR = os.path.abspath("../")
DATA_DIR = os.path.join(ROOT_DIR, "Videos")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "mask_rcnn_stones.h5")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DETECTIONS_DIR = os.path.join(ROOT_DIR, "detections")
CLASS_NAMES = ["Background", "Stone"]

parser = ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()
VIDEO = args.video

VIDEO_PATH = os.path.join(DATA_DIR, VIDEO)

if not os.path.exists(LOGS_DIR):
  os.mkdir(LOGS_DIR)

if not os.path.exists(DETECTIONS_DIR):
  os.mkdir(DETECTIONS_DIR)

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config, 
                          model_dir=LOGS_DIR)
model.load_weights(CUSTOM_MODEL_PATH, by_name=True)

video_pointer = cv2.VideoCapture(VIDEO_PATH)
frames_per_second = video_pointer.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAPS_PROPS_FPS) :", format(frames_per_second))

frames = []
frame_count = 0
signals = 0
short_time_detections = 0
break_in_detection = 0
count_stones = 0
time_start = time.time()
while True:
  success, frame = video_pointer.read()
  if not success:
    break
  frame_count += 1
  if DEBUG:
    print("Frame ", frame_count)
  detected = False
  result = model.detect([frame], verbose=0)
  result = result[0]
  if len(result["scores"]) > 0:
    if DEBUG:
      print("DETECTION WITH SCORES: ", result["scores"])
    detected = True
  elif DEBUG:
    print("NO DETECTIONS")
  frame = apply_instances(
    frame,
    result['rois'],
    result['masks'],
    result['class_ids'],
    CLASS_NAMES,
    result["scores"]
  )
  frames.append(frame)

  if detected:
    break_in_detection = 0
    count_stones += 1
  if count_stones == 30:
    signals += 1
    count_stones = 0
  if count_stones != 0 and not detected:
    break_in_detection += 1
    if break_in_detection > 5:
      short_time_detections += count_stones
      count_stones = 0

full_time = time.time() - time_start
fps = frame_count / full_time

video_pointer.release()
output_video = os.path.join(DETECTIONS_DIR, VIDEO)
make_video(output_video, frames)

print("STOP signals: ", signals)
print("Short time detections: ", short_time_detections)
print("Time: ", full_time, " s")
print("FPS: ", fps)