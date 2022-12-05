#!/usr/bin/env python3
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("start", type=int)
parser.add_argument("stop", type=int)
args = parser.parse_args()
path  = args.path 
start = args.start 
stop  = args.stop
all_images = os.listdir(path)
filenames = []
for image in all_images:
  for i in range(start, stop + 1):
    if image == str(i) + "_undetected.jpg":
      os.rename(os.path.join(path, image), os.path.join(path, str(i) + "_detected.jpg"))
  