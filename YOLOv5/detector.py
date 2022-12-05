#!/usr/bin/env python3

import os
import subprocess
from argparse import ArgumentParser
import sys
import time

DEBUG = False

ROOT_DIR = os.path.abspath(".")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
STONE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov5_stones.pt")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")
parser = ArgumentParser()
parser.add_argument("input_video")

args = parser.parse_args()
video = args.input_video
video = os.path.join(VIDEOS_DIR, video)

COMMAND = [
  'python3',
  'detect.py',
  '--weights',
  STONE_MODEL_PATH,
  '--source',
  video
]

time_start = time.time()
p = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, bufsize=1)

counter = 0
count_frames = 0
short_time_detections = 0
signals = 0
break_in_detection = 0
while True:
  line = p.stdout.readline()
  if line:
    line = line.decode('utf-8')
    output = line.split(' ')
    if DEBUG:
      print(output)
    if len(output) > 6 and output[0] == 'video':
      count_frames += 1
      if 'stone' in output[6]:
        counter += 1 
        break_in_detection = 0
      else:
        if counter != 0:
          break_in_detection += 1
          if break_in_detection > 5:
            short_time_detections += counter
            counter = 0

      if counter == 30:
        signals += 1

    sys.stdout.flush()
  if p.poll() is not None:
    break
  
full_time = time.time() - time_start
fps = count_frames / full_time
print("STOP signals: ", signals)
print("Short time detections: ", short_time_detections)
print("Time: ", full_time, " s")
print("Frames per second: ", fps, " s")

p.wait()