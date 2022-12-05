#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.abspath(".")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")
YOLO_DETECTIONS_DIR = os.path.join(ROOT_DIR, "runs/detect")
DETECTIONS_DIR = os.path.join(ROOT_DIR, "detections")

if not os.path.exists(DETECTIONS_DIR):
  os.mkdir(DETECTIONS_DIR)

for video_name in os.listdir(VIDEOS_DIR):
  COMMAND = [
    "./detector.py",
    video_name
  ]
  p = subprocess.Popen(COMMAND, stdout = subprocess.PIPE, bufsize=1)
  print("Running detector.py for video file ", video_name)
  print("---------------------------------------")
  while True:
    line = p.stdout.readline()
    if line:
      line = line.decode('utf-8')
      print(line)
      sys.stdout.flush()
    if p.poll() is not None:
      break
  print("---------------------------------------")
  p.wait()
  for dir in os.listdir(YOLO_DETECTIONS_DIR):
    dir = os.path.join(YOLO_DETECTIONS_DIR, dir)
    for file in os.listdir(dir):
      file = os.path.join(dir, file)
      shutil.copy(file, DETECTIONS_DIR)
      os.remove(file)