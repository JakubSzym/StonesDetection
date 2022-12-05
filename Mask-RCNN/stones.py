from config import Config
import utils
import numpy as np
import os
import json
import skimage

class StonesConfig(Config):
  NAME = "stones"
  GPU_COUNT = 1
  IMAGES_PER_GPU = 2
  NUM_CLASSES = 2
  TRAIN_ROIS_PER_IMAGE = 32
  STEPS_PER_EPOCH = 100
  VALIDATION_STEPS = 5
  MAX_GT_INSTANCES = 5
  USE_MINI_MASK = False
  DETECTION_MIN_CONFIDENCE = 0.9

class StonesDataset(utils.Dataset):
  def load_custom(self, dataset_dir, subset):
    self.add_class("object", 1, "stone")

    assert subset in ["train", "test", "valid"]
    dataset_dir = os.path.join(dataset_dir, subset)

    json_info = json.load(open(os.path.join(dataset_dir, "_annotations.coco.json")))

    annotations = {}
    for annotation in json_info["annotations"]:
      image_id = annotation["image_id"]
      if image_id not in annotations:
        annotations[image_id] = []
      annotations[image_id].append(annotations)
    
    images = {}
    for image_info in json_info["images"]:
      image_id = image_info["id"]
      if image_id in images:
        print("INFO: Skipping image to avoid duplications")
      else:
        filename = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        path = os.path.abspath(os.path.join(dataset_dir, filename))
        image_annotations = annotations[image_id]

        self.add_image(
          source="object",
          image_id=image_id,
          path=path,
          width=width,
          heigth=height,
          annotations=image_annotations
        )

  def load_mask(self, image_id):
    info = self.image_info[image_id]
    if info["source"] != "object":
      return super(self.__class__, self).load_mask(image_id)
    num_ids = []
    mask = np.zeros([info["height"], info["width"], len(info["annotations"])],
                      dtype=np.uint8)
    for i, a in enumerate(info["annotations"]):
      num_ids.append(a["category_id"])
      x, y, w, h = a["bbox"]
      start = (y,x)
      extent = (int(h), int(w))
      rr, cc = skimage.draw.rectangle(start=start, extent=extent, shape=[info["height"], info["width"]])
      mask[rr, cc, i] =  1
    
    num_ids = np.array(num_ids, dtype=np.int32)
    return mask, num_ids

  def image_reference(self, image_id):
    info = self.image_info[image_id]
    if info["source"] == "object":
      return info["path"]
    else:
      return super().image_reference(image_id)