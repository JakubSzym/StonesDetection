#!/usr/bin/env python3

import os
import utils
import model as modellib
from model import log
from stones import StonesConfig, StonesDataset

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
  utils.download_trained_weights(COCO_MODEL_PATH)

config = StonesConfig()
config.display()

dataset_train = StonesDataset()
dataset_train.load_custom(DATA_DIR, "train")
dataset_train.prepare()

dataset_test = StonesDataset()
dataset_test.load_custom(DATA_DIR, "test")
dataset_test.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_test,
              learning_rate=config.LEARNING_RATE,
              epochs=15,
              layers="heads")

model_path = os.path.join(MODEL_DIR, "mask_rcnn_stones.h5")
model.keras_model.save_weights(model_path)