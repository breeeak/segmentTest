import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler


class RPNHead(nn.Module):  # RPN部分的网络结构
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)  # cls中先做一个3x3的卷积
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)  # 接着通过一个1x1的卷积，获得每一个anchors对应的cls值
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)  # bbox_pred是通过一个1x1的卷积，获得4*num_anchors维度的对于原有anchor的调整

        for l in self.children():  # 对RPN中的参数进行初始化
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        x = F.relu(self.conv(x))  # 前两个有关cls，代表RPN图中的上一行
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)  # 这一行有关bbox_pred，代表RPN图中的下一行
        return logits, bbox_reg
    

class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator  # 用于产生作为基础的anchor的位置
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)  # 返回一张表说明该anchors是正样本还是负样本
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)  # 根据label（上一个函数的返回结果），和所需样本数及正样本比重，返回正样本和负样本的index
        self.box_coder = BoxCoder(reg_weights)  # 该函数用于将RPN的bbox预测值和anchor_generator编解码成实际的bbox的位置
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
                
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)  # 输出的nms数量，等于目标的数目和设定的nms数量的最小值
        top_n_idx = objectness.topk(pre_nms_top_n)[1]  # 找到objectness中最大的pre_nms_top_n个的index
        score = objectness[top_n_idx]  # 找出最高这几个的cls值
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])  # 解码为相对于pred_bbox_delta的bbox的位置
        
        proposal, score = process_box(proposal, score, image_shape, self.min_size)  # 该函数的作用在于使得bbox不超过图片的范围，且删除一些宽高小于min_size的bbox
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n]  # 实现了非极大值抑制，最多取post_nms_top_n个
        proposal = proposal[keep]
        return proposal
    
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)  # 计算gt与预先设定的anchor之间的iou
        label, matched_idx = self.proposal_matcher(iou)  # 返回是属于正样本还是负样本
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)  # 根据采样总数和正样本比例，找出正负样本的index
        idx = torch.cat((pos_idx, neg_idx))  # 获得总的用来训练的index
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])  # 将gt的位置编码为与模型输出的位置一致
        
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])  # 计算cls的loss
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()  # 用l1范数计算bbox部分的loss

        return objectness_loss, box_loss
        
    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['boxes']
        anchor = self.anchor_generator(feature, image_shape)  # 计算得到anchor的相对位置
        
        objectness, pred_bbox_delta = self.head(feature)  # 输入RPN的网络中，获得置信度和预测框
        objectness = objectness.permute(0, 2, 3, 1).flatten()  # 将cls的那一通道移到最后一位
        pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)  # 将bbox的那一通道移到最后一位

        proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)
        if self.training:  # 如果是train阶段，计算此时的loss
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}