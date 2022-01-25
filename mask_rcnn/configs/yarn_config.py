# -*- coding: utf-8 -*-
# @Time    : 2022/1/21 19:30
# @Author  : Marshall
# @FileName: yarn_config.py

from configs.config import Config
import os
# TODO need change to my dataset


class YarnConfig(Config):
    """Configuration for training on yarn dataset， it is similar to COCO
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "yarn1"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    DEFAULT_DATASET_YEAR = "2014"

    DEFAULT_DATASET_PATH = "E:\\1_dataset\\common_dataset\\COCO"

    # 声明类别，尽量保持
    CLASS_NAMES = ["weft", "warp"]
    # 数据集路径
    ANN_ROOT = os.path.join(DEFAULT_DATASET_PATH, 'annotations')
    TRAIN_PATH = os.path.join(DEFAULT_DATASET_PATH, 'train_set')
    VAL_PATH = os.path.join(DEFAULT_DATASET_PATH, 'test_set')
    TRAIN_JSON = os.path.join(DEFAULT_DATASET_PATH, 'train_annotation.json')
    VAL_JSON = os.path.join(DEFAULT_DATASET_PATH, 'test_annotation.json')
