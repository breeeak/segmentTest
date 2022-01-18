import torch
import torch.nn as nn
import config as cfg
import numpy as np
from torchvision.ops import nms
from rpn.generate_anchors import generate_anchors
from rpn.bbox_transform import bbox_transform_inv, clip_boxes


class _ProposalLayer(nn.Module):
    #_ProposalLayer输入：RPN的分类和回归两个分支的结果（置信度、位置调整参数）、im_info（记录原图尺寸和stride）用于生成anchor box的scales, ratios两参数
    #输出：NMS后的top2000 anchor box
    def     __init__(self, feature_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feature_stride
        self._anchors = torch.from_numpy(generate_anchors(feature_stride=16,
                                                          scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        ##1、inpu中记录scores、bbox_deltas、im_info、is_training四个值
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        is_training = input[3]

        if is_training:
            pre_nms_topN = cfg.train_rpn_pre_nms_top_N   #12000
            post_nms_topN = cfg.train_rpn_post_nms_top_N #2000
            nms_thresh = cfg.rpn_nms_thresh              #0.7
        else:
            pre_nms_topN = cfg.test_rpn_post_nms_top_N   #6000
            post_nms_topN = cfg.test_rpn_post_nms_top_N  #300
            nms_thresh = cfg.rpn_nms_thresh              #0.7

        batch_size = bbox_deltas.size(0)
        ##2、从RPN的scores分支中提取feature map的高和宽
        feat_height, feat_width = scores.size(2), scores.size(3)
        ##3、使用np.meshgrid方法生成生成网格shifts
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()
        ##4、生成anchor  # A为Anchor的数量，也就是3*3=9,K为feature map的宽*高，如50*38=1900（每张图都可能不一样）
        A = self._num_anchors
        K = shifts.size(0)
        #生成anchor，数量为K*A
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        #转化为batch_size,num,4
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        #将bbox_deltas（RPN输出的anchor box的位置调整参数）调整为anchors相同的shape
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        #print(im_info)
        # 将超出范围的候选框给裁剪使其不超过图像范围
        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]
            #选取前pre_nms_topN
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            #nms
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)
            #选取nms之后的前post_nms_topN个
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i   #属于哪个batch
            output[i, :num_proposal, 1:] = proposals_single  #候选框坐标

        return output


    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))

        return keep


