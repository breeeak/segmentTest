# -*- coding: utf-8 -*-
# @Time    : 2022/1/25 17:06
# @Author  : Marshall
# @FileName: evaluate.py
from configs.coco_config import CocoConfig
from datasets.CocoDataset import CocoDataset
from engines.engine import Engine


if __name__ == '__main__':
    config = CocoConfig()

    model = Engine(config=config)

    if config.GPU_COUNT:
        model = model.cuda()
    # Set weights path, default find last trained weights
    model_path = model.find_last()[1]
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco(config.DEFAULT_DATASET_PATH, "test", year=config.DEFAULT_DATASET_YEAR)
    dataset_val.prepare()

    # Images to use for evaluation
    img_num = 500
    print("Running COCO evaluation on {} images.".format(img_num))
    dataset_val.evaluate_coco(model, dataset_val, coco, "bbox", limit=img_num)
    dataset_val.evaluate_coco(model, dataset_val, coco, "segm", limit=img_num)
