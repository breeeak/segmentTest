# -*- coding: utf-8 -*-
# @Time    : 2022/1/21 19:04
# @Author  : Marshall
# @FileName: train.py
import argparse
from yarn_config import YarnConfig
from modellib import MaskRCNN
from datasets.YarnDataset import YarnDataset
from engines.engine import Engine

if __name__ == '__main__':

    config = YarnConfig()

    model = Engine(config=config)

    if config.GPU_COUNT:
        model = model.cuda()

    dataset_train = YarnDataset()
    dataset_val = YarnDataset()

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



