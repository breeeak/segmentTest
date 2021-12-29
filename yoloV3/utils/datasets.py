# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov3_config_voc as cfg
import cv2
import numpy as np
import random
# from . import data_augment as dataAug
# from . import tools

import utils.data_augment as dataAug
import utils.tools as tools
import utils.visualize as vis


class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        """
        需要先准备数据集，参见VOC.py，生成txt文件
        anno_file_type  "You must choice one of the 'train' or 'test' for anno_type parameter", 用来拼接路径数据的，不需要传入具体的路径了
        :param anno_file_type:
        :param img_size:
        """
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))     # 生成class一一对应id的字典，按顺序来的,跟voc处理的对应，反正所有的都统一
        self.__annotations = self.__load_annotations(anno_file_type)      # 加载数据集，txt中的每一行，即每一个图片

    def __len__(self):
        # 有这个方法才可以len
        return  len(self.__annotations)

    def __getitem__(self, item):
        # 有这个方法才可以enumerate
        # 先解析annotation行，输出原始图片与bbox
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
        # 这里是再从数据集中随机选一张图片
        item_mix = random.randint(0, len(self.__annotations)-1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)



        # 选出的两张图片进行概率随机，拼接混合，相当于两张图片叠加; box第二个维数增加1，表示每个图片的混合比例，第一个维数表示有几个对象
        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)     # 混合这些框，利用拼接等
        del img_org, bboxes_org, img_mix, bboxes_mix


        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)   # 对框根据锚进行处理

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes




    def __load_annotations(self, anno_type):

        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(cfg.PROJECT_PATH, 'data', anno_type+"_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x:len(x)>0, f.readlines()))        # 读取有内容的行
        assert len(annotations)>0, "No images found in {}".format(anno_path)

        return annotations  # 返回的是txt中的每一行

    def __parse_annotation(self, annotation):
        """
        返回一张图片和原始bbox，图片不一定经过数据增强
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        # 这些是随机数据增强，不一定发生,box一起变化会，这些都对目标识别无影响
        img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        # 最终变成规定的大小
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):
        """
        生成不同尺度下的标签
        1.先("xyxy") to "xywh"，然后根据stride缩放bbox，生成gtbox
        2. 依次计算各scale层锚框与gtbox之间的iou,并选择最大的，如果所有检测层的质量小于0.3，则选择最大
        3. 生成onehot的分类标签

        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.
        bbox的总数可能比实际的要多，因为相同的GT可能被分配到多个层的检测。

        """

        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6+self.num_classes))
                                                                      for i in range(3)]
        # 将内里的第5列设为1，mix先默认给1。
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)
            # 这里是对bbox进行scale，换算成3个比例下的尺寸，size=3，4, 其实主要保留中心的变化
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):      # 这里写成固定值了，是按照scale数来，可以改进？ 但是要大改
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                # anchors 前两维是中心点位置，这里向下取整再加0.5做补偿， 0那层是52*52,1是26*26，所以没错
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                # 用给出的先验框大小来替代标注的框的大小
                anchors_xywh[:, 2:4] = anchors[i]
                # 计算iou, scale后的bbox_xywh_scaled,与anchors_xywh，大小不一样，中心一样
                # 输入的维度数要相等，所以要在一开始扩充维度，就是说可以一个真实的框与多个生成框，分别计算IOU
                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3      # TODO 这里认为iou>0.3就是一个正值，是否可以设置成一个参数来进行调节？ 不是说是最大iou作为正例吗？ 大于0.7?

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    # 这里使用的是原始的bbox_xywh用来保存做label，使用scaled来计算iou，是用来确定对应的特征图的位置的
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    # 这里用来储存所有的原始的bbox_xywh,并且这里初始化了一个150大小，就是最多150个目标框，再大就会覆盖第一个，
                    # 这里是用来记录网络输入的位置，用来训练的x
                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
            # 如果一个大于规定值的iou的都没有，使用最大的那个iou作为正值，iou全都保存下来了，所以对3取余就是哪个尺度下的
            # 保证要有正例
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:       # enumerate 默认批量就是1
                labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
                                         label_lbbox.reshape(-1, 26)], axis=0)

                labels_mask = labels[..., 4]>0      # 大于0的才是真值
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                                        axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
