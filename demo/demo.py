# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from pathlib import Path

# constants
def save_result_to_txt(txt_save_path,prediction):

    file = open(txt_save_path,'w')
    classes = prediction['instances'].pred_classes
    boxes = prediction['instances'].pred_boxes.tensor

    for i in range(len(classes)):
        if classes[i]==0:
            xmin = str(int(boxes[i][0]))
            ymin = str(int(boxes[i][1]))
            xmax = str(int(boxes[i][2]))
            ymax = str(int(boxes[i][3]))

            file.writelines(xmin+','+ymin+','+xmax+','+ymax+',')
            file.writelines('\r\n')
    file.close()

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file("./configs/ocr/icdar2013_101_FPN.yaml")
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    Path("/content/detection_result/").mkdir(parents=True, exist_ok = True)

    cfg = setup_cfg()
    demo = VisualizationDemo(cfg)
    imgs = os.listdir("/content/test/")
    for path in imgs:
        fullpath = "/content/test/" + path
        img = read_image(fullpath, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        save_result_to_txt("/content/detection_result/" + path.split(".")[0] + ".txt", predictions)
        visualized_output.save("/content/detection_result/" + path)

