import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride


    def forward(self, p):   # p是Bachsize*75*52*52
        bs, nG = p.shape[0], p.shape[-1]
        # 先进行reshape结构，再调整通道数
        # bs,anchors_num,bbox+类别数,宽，高; 再调整顺序为：bs,宽，高，anchors_num，bbox+类别数(25)
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)     # 这里最后为什么要两次reshape

        p_de = self.__decode(p.clone())

        return (p, p_de)


    def __decode(self, p):
        """
        生成锚框，调整先验框
        :param p:
        :return:
        """
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)     # anchors无需放大，乘以stride，因为前面计算iou的时候用的就是这样的，如果看到比较大的就需要乘stride
        # 提取前两维作为调整参数x,y;因为之前是用中心点来作为的anchor值。现在如果从左上角来看，中心点刚刚好是调整值。
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        # 提取宽高的调整参数
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        # 提取置信度，有无目标
        conv_raw_conf = p[:, :, :, :, 4:5]
        # 提取onehot类别
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)    # 这里就相当于是一个网格，y值 行一样
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)    # x列一样
        grid_xy = torch.stack([x, y], dim=-1)   # 最后是一个52*52的网格坐标
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)   # 这里三张网，对应三个不同大小先验框，还有batch_size，的一个大网格
        # 中心点调整参数使用sigmoid进行处理，需要×以stride  grid即是52*52的网格
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        # 宽高是进行了指数运算进行调整 乘以anchor对应尺寸，这里对应的都是论文原文中的讲解
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)    # 这里是置信度输出
        pred_prob = torch.sigmoid(conv_raw_prob)    # 这里是类别输出，对应原文没有使用sigmoid
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        # 如果是预测的话，就把类别预测放在最前面
        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
