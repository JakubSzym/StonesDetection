#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.abspath(".")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")

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