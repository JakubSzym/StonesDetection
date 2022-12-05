#!/usr/bin/env python3

import os
import cv2
from argparse import ArgumentParser
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import numpy as np
import time

DEBUG = False

def make_boxes(frame, boxes, scores, labels):
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < MIN_CONFIDENCE:
      break

    b = np.array(box).astype(int)
    (startX, startY, endX, endY) = b
    frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    frame = cv2.putText(frame, caption, (startX, startY - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
  return frame

def make_video(output_video, images):
  example_frame = images[0]
  height, width = example_frame.shape[:2]
  video_writer = cv2.VideoWriter(output_video, 0, 25, (width, height))
  for image in images:
    video_writer.write(image)
  video_writer.release()

MIN_CONFIDENCE = 0.5

ROOT_DIR = os.path.abspath(".")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")
MODEL_PATH = os.path.join(ROOT_DIR, "retinanet_stones.h5")
DETECTIONS_DIR = os.path.join(ROOT_DIR, "detections")

if not os.path.exists(DETECTIONS_DIR):
  os.mkdir(DETECTIONS_DIR)

parser = ArgumentParser()
parser.add_argument("input_video")
args = parser.parse_args()
video_name = args.input_video

VIDEO_PATH = os.path.join(VIDEOS_DIR, video_name)

model = models.load_model(MODEL_PATH, backbone_name='resnet50')
labels_to_names = {0: 'stone'}

video_pointer = cv2.VideoCapture(VIDEO_PATH)
frames = []
count = 0
count_stones = 0
signals = 0
short_time_detections = 0
break_in_detection = 0

time_start = time.time()

while True:
  success, frame = video_pointer.read()
  if not success:
    break
  count += 1
  if DEBUG:
    print("Frame: ", count)
  frame = np.asarray(frame)
  frame = frame[:, :, ::-1].copy()
  draw = frame.copy()
  frame = preprocess_image(frame)
  frame, scale = resize_image(frame)
  frame = np.expand_dims(frame, axis=0)
  boxes, scores, labels = model.predict_on_batch(frame)
  boxes /= scale
  detected = False
  for score in scores[0]:
    if score >= MIN_CONFIDENCE:
      detected = True
      break
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
  draw = make_boxes(draw, boxes, scores, labels)
  frames.append(draw)

full_time = time_start - time.time()
fps = count / full_time
OUT_VIDEO_PATH = os.path.join(DETECTIONS_DIR, video_name)
make_video(OUT_VIDEO_PATH, frames)

print("STOP signals: ", signals)
print("Short time detections: ", short_time_detections)
print("Time: ", full_time, " s")
print("FPS: ", fps)