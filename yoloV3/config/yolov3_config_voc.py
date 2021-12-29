# coding=utf-8

# 想要跑自己的模型，可以做成VOC数据集样式，然后修改类别信息，先验框这里可以先不用调，一般都可以跑通，只要是做成VOC格式图片大小，当然也可调整图片大小
# VOC的目标检测标签在annotation下
# project
DATA_PATH = "E:/1_dataset/common_dataset/VOC/data/VOC"
# 项目路径
PROJECT_PATH = "D:/3_Research/2_Learning/Codes/Python/segmentTest/yoloV3"

# 类别信息
DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}

# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj  52*52 面积下的 先验框宽和搞
                    # 先验框的宽和高，这里是根据数据集聚类生成的，不是比例，是不同尺度下的像素宽和高
                    # 先验框这里可以先不用调，一般都可以跑通，只要是做成VOC格式
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj  26*26
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj  13*13
         "STRIDES":[8, 16, 32],     # 每个特征图对应的大小，1/8,1/16,1/32,换算出来就是416/8=52
         "ANCHORS_PER_SCLAE":3      # 每个anchor生成几个框
         }

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    #下面是学习率相关, WARMUP是让学习率不要一下子太高
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}


# test
TEST = {
        "TEST_IMG_SIZE":416,
        "BATCH_SIZE":4,
        "NUMBER_WORKERS":2,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }
