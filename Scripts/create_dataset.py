#!/usr/bin/env python3
import cv2
import os
import argparse

IMG_SIZE = 800

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")

argv = parser.parse_args()

input_path = argv.input_path
output_path = argv.output_path

counter = 0
for file_name in os.listdir(input_path):
  video_pointer = cv2.VideoCapture(os.path.join(input_path, file_name))
  success, img = video_pointer.read()
  while success:
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(os.path.join(output_path, str(counter) + "_undetected.jpg"), img)
    success, img = video_pointer.read()
    counter += 1
