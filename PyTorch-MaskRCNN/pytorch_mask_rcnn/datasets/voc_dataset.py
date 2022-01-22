import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms

from .generalized_dataset import GeneralizedDataset


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)

def target_to_coco_ann(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    area = boxes[:, 2] * boxes[:, 3]
    area = area.tolist()
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    anns = []
    for i, rle in enumerate(rles):
        anns.append(
            {
                'image_id': image_id,
                'id': i,
                'category_id': labels[i],
                'segmentation': rle,
                'bbox': boxes[i],
                'area': area[i],
                'iscrowd': 0,
            }
        )
    return anns     


class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        # instances segmentation task
        id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split))  # 得到训练集的数据的id
        self.ids = [id_.strip() for id_ in open(id_file)]  # 并存在`self.ids`中
        self.id_compare_fn = lambda x: int(x.replace("_", ""))  # 替换掉数字中的`_`为``
        
        self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split))  # 得到ann文件的地址
        self._coco = None
        
        self.classes = VOC_CLASSES  # VOC中所存在的几种类
        # resutls' labels convert to annotation labels
        self.ann_labels = {self.classes.index(n): i for i, n in enumerate(self.classes)}
        
        checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self.make_aspect_ratios()  # 计算每一个图片的宽高比
            self.check_dataset(checked_id_file)  # 判断一个图片的信息中，是否boxes，labels，masks都齐了,voc中好像有个图像没有 2009_005069
            
    def make_aspect_ratios(self):
        self._aspect_ratios = []
        for img_id in self.ids:
            anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
            size = anno.findall("size")[0]
            width = size.find("width").text
            height = size.find("height").text
            ar = int(width) / int(height)
            self._aspect_ratios.append(ar)

    def get_image(self, img_id):  # 根据ID读入图片
        image = Image.open(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id)))
        return image.convert("RGB")
        
    def get_target(self, img_id):  # 根据ID获得训练目标，包括boxes，标签和masks
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))  # mask的数值来自于图片
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()  # 找到mask中有哪几种不同的数值
        uni = uni[(uni > 0) & (uni < 1)]  # 其中在0和1之间的数值分别代表一个mask
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)  # 将图片中等于uni的部分分别转为值为0,1的图片        类别，宽，高 格式，每一个类别一个通道
        
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))  # 对应的xml文件中读入
        boxes = []
        labels = []
        for obj in anno.findall("object"):  # 读入每一个图片的boxes和classes信息
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = self.classes.index(name)

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)  # 转成tensor的数据类型
        labels = torch.tensor(labels)  # 转成tensor的数据类型

        img_id = torch.tensor([self.ids.index(img_id)])
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)  # 将所有的信息封装成dict，然后返回
        return target
    
    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO
            self.convert_to_coco_format()
            self._coco = COCO(self.ann_file)
        return self._coco
    
    def convert_to_coco_format(self, overwrite=False):
        if overwrite or not os.path.exists(self.ann_file):
            
            
            print("Generating COCO-style annotations...")
            voc_dataset = VOCDataset(self.data_dir, self.split, True)
            instances = defaultdict(list)
            instances["categories"] = [{"id": i, "name": n} for i, n in enumerate(voc_dataset.classes)]

            ann_id_start = 0
            for image, target in voc_dataset:
                image_id = target["image_id"].item()

                filename = voc_dataset.ids[image_id] + ".jpg"
                h, w = image.shape[-2:]
                img = {"id": image_id, "file_name": filename, "height": h, "width": w}
                instances["images"].append(img)

                anns = target_to_coco_ann(target)
                for ann in anns:
                    ann["id"] += ann_id_start
                    instances["annotations"].append(ann)
                ann_id_start += len(anns)

            json.dump(instances, open(self.ann_file, "w"))
            print("Created successfully: {}".format(self.ann_file))
        
  