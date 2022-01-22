# -*- coding: utf-8 -*-
# @Time    : 2022/1/21 19:30
# @Author  : Marshall
# @FileName: yarn_config.py

from configs.config import Config

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

    DEFAULT_DATASET_PATH = "data"
