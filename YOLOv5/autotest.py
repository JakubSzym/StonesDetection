#!/usr/bin/env python3

###############################
# Author: Jakub Szymkowiak    #
# Title:
# Year: 2022                  #
# Module: autotest.py         #
###############################

import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(".")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")

for video in os.listdir(VIDEOS_DIR):
  COMMAND = [
    './detector.py',
    video
  ]
  p = subprocess.Popen(COMMAND, stdout = subprocess.PIPE, bufsize=1)
  while True:
    line = p.stdout.readline()
    if line:
      line = line.decode('utf-8')
      print(line)
      sys.stdout.flush()
    if p.poll() is not None:
      break
  p.wait()