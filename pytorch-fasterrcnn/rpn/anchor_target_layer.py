import torch
import torch.nn as nn
import numpy as np
import config as cfg
from rpn.generate_anchors import generate_anchors
from rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class _AnchorTargetLayer(nn.Module):
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        #定义每个anchor点上待生成的9个基础anchor box
        self._anchors = torch.from_numpy(
            generate_anchors(self._feat_stride, scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        self._allowed_border = 0  # allow boxes to sit over the edge by a small amount

    def forward(self, input):

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        batch_size = gt_boxes.size(0)

        #生成网络
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()  #深拷贝

        A = self._num_anchors
        K = shifts.size(0)  #number of ceil

        self._anchors = self._anchors.type_as(gt_boxes)  # move to specific gpu.
        #基础anchor box和网络点运算，生成每张图的anchor boxes（数量为 w*h*9）
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        #reshape为（w*h*9，w）
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)
        # 计算没有超出边界的锚点框，keep为True False，True的box将保留
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))
        # 返回非零元素的索引
        inds_inside = torch.nonzero(keep).view(-1)

        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        #计算anchor boxes 与gt_box的交并比
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        #print('anchor overlaps', overlaps)
        # 每行的最大值和所在的坐标
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # 每列的最大值所在的坐标
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        #小于0.3的label为0，即背景bg
        labels[max_overlaps < cfg.rpn_neg_iou_thr] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
        #大于0.7的label为1，即前景fg
        labels[max_overlaps >= cfg.rpn_pos_iou_thr] = 1

        num_fg = int(cfg.anchor_batch_size * 0.5)
        # 统计fg和bg的数量
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have  too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.anchor_batch_size - torch.sum((labels == 1).int(), 1)[i]
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        #每个anchor跟踪它对应的最大IOU的gt框进行回归
        bbox_targets = _compute_targets_batch(anchors,
                                              gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        #计算回归损失时，乘以这个矩阵就可以屏蔽0和-1的anchor
        bbox_inside_weights[labels == 1] = cfg.rpn_bbox_inside_weight[0]

        num_examples = torch.sum(labels[i] >= 0)
        positive_weights = 1.0 / num_examples.item()
        negative_weights = 1.0 / num_examples.item()

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, feat_height, feat_width, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * feat_height, feat_width)   #这里这样操作一下有什么意义吗
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, feat_height, feat_width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, feat_height, feat_width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_inside_weights)
        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                              4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, feat_height, feat_width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        outputs.append(bbox_outside_weights)

        return outputs


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])

