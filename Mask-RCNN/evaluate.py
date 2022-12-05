#!/usr/bin/env python3
import os
import numpy as np

import utils

from stones import StonesConfig, StonesDataset

ROOT_DIR = os.path.abspath("../")

import model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

CUSTOM_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_stones.h5")

class InferenceConfig(StonesConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

print("Loading weights from ", CUSTOM_MODEL_PATH)
model.load_weights(CUSTOM_MODEL_PATH, by_name=True)

dataset_test = StonesDataset()
dataset_test.load_custom("../data", "valid")
dataset_test.prepare()

APs = []
F1_values = []

for image_id in dataset_test.image_ids:

  image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config,
                          image_id)
  molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

  results = model.detect([image], verbose=0)
  r = results[0]

  AP, precisions, recalls, overlaps =\
    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                      r['rois'], r['class_ids'], r['scores'], r['masks'])
  APs.append(AP)
  F1_values.append(2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)))

print("mAP: ", np.mean(APs))
print("F1: ", np.mean(F1_values))