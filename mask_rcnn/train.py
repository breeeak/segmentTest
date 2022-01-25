# -*- coding: utf-8 -*-
# @Time    : 2022/1/21 19:04
# @Author  : Marshall
# @FileName: train.py
import argparse
from configs.coco_config import CocoConfig
from datasets.CocoDataset import CocoDataset
from engines.engine import Engine

import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)     # cpu 种子
    torch.cuda.manual_seed_all(seed)    # gpu种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True   # cudnn


if __name__ == '__main__':
    setup_seed(777)
    config = CocoConfig()

    model = Engine(config=config)

    if config.GPU_COUNT:
        model = model.cuda()

    model_path = ""
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = CocoDataset()
    dataset_train.load_coco(config.DEFAULT_DATASET_PATH, "train", year=config.DEFAULT_DATASET_YEAR)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(config.DEFAULT_DATASET_PATH, "val", year=config.DEFAULT_DATASET_YEAR)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE,
                      epochs=40,
                      layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE,
                      epochs=120,
                      layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE / 10,
                      epochs=160,
                      layers='all')



