import torch
import torch.nn.functional as F
from torch import nn

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)  # 类别的loss用交叉熵来计算

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]  # 由于`select_training_samples`中pos的index接在前面，所以前几个就代表pos的
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N  # 对正样本计算bbox预测的loss

    return classifier_loss, box_reg_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)  # 其中的.to用于统一数据类型
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]  # 调用ROIAlign获得目标的mask

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)  # 计算mask部分的loss
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool  # 对应之前的`box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)`
        self.box_predictor = box_predictor  # 对应`FastRCNNPredictor(in_channels, mid_channels, num_classes)`
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)  # 返回一张表说明该anchors是正样本还是负样本, 这里是false 说明大于iou就认为是一个正样本，matched_idx就一定有目标
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)  # 根据label（上一个函数的返回结果），和所需样本数及正样本比重，返回正样本和负样本的index
        self.box_coder = BoxCoder(reg_weights)  # 该函数用于将RPN的bbox预测值和anchor_generator编解码成实际的bbox的位置
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        proposal = torch.cat((proposal, gt_box))        # proposal 是nms得到的估计有目标的位置，网络的proposal ， rpn中计算loss是计算anchor的 为何加上真实值？ 第二个网络，proposal就相当于已知值了，当做标签来使用了。所以要加上真实值
        
        iou = box_iou(gt_box, proposal)  # # 计算gt与proposal之间的iou
        pos_neg_label, matched_idx = self.proposal_matcher(iou)  # 返回是属于正样本还是负样本
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)  # 根据采样总数和正样本比例，找出正负样本的index
        idx = torch.cat((pos_idx, neg_idx))  # 获得总的用来训练的index
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])  # 将gt的位置编码为与模型输出的位置一致
        proposal = proposal[idx]  # 得到所有用于训练的proposal
        matched_idx = matched_idx[idx]  # 得到所有用于训练的matched_idx
        label = gt_label[matched_idx]  # 得到所有用于训练的label， 这里对应具体的目标了，而不再是前景背景了
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0
        
        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)  # 类别做softmax
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):  # 一类一类来，排除第一类（背景）
            score, box_delta = pred_score[:, l], box_regression[:, l]  # 选出那一类对应的

            keep = score >= self.score_thresh  # 只有预测值高于score_thresh时才会认为有那一类
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]  # 抽出此时对应的box，score，box
            box = self.box_coder.decode(box_delta, box)  # 解码box_delta为绝对位置
            
            box, score = process_box(box, score, image_shape, self.min_size)  # 该函数的作用在于使得bbox不超过图片的范围，且删除一些宽高小于min_size的bbox
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]  # 实现了非极大值抑制，最多取num_detections个
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))  # 存入所有结果并输出
        return results
    
    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)  # 根据规则筛选出用于训练的目标
        
        box_feature = self.box_roi_pool(feature, proposal, image_shape)  # 用ROIAlign得到大小为[512,256,7,7]的特征图,之所以需要这样是因为后面有全连接，预测类别，必须保证前面的尺寸固定。ROIAlign可以看成是一种改进了的Maxpooling 采用了双线性插值。具体参考知乎
        class_logit, box_regression = self.box_predictor(box_feature)  # 得到分类结果和bbox预测结果， fasterRCNN 使用全连接了。#
        
        result, losses = {}, {}
        if self.training:       # TODO Loss是如何计算的 box部分 类别预测部分
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)  # 返回得到cls和bbox的loss
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)  # 用于推断的部分
            
        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                
                mask_proposal = proposal[:num_pos]  # 先把对应正样本的位置准备好
                pos_matched_idx = matched_idx[:num_pos]  # 先把对应正样本的index也准备好
                mask_label = label[:num_pos]  # 先把对应正样本的label也准备好
                
                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                
                if mask_proposal.shape[0] == 0:  # 如果没有mask_proposal，则直接返回0
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:  # 这里是inference部分，`result['boxes']`是`self.fastrcnn_inference`的bbox输出
                mask_proposal = result['boxes']
                
                if mask_proposal.shape[0] == 0:  # 如果没有预测框，则直接返回0
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)  # 对应第二个ROIAlign，得到[?,256,14,14]
            mask_logit = self.mask_predictor(mask_feature)  # 输入分割网络，得到[?,cls+1,28,28]
            
            if self.training:
                gt_mask = target['masks']  # 得到对应的mask（原始）
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]  # 按照label的顺序排列mask

                mask_prob = mask_logit.sigmoid()  # 并做sigmoid
                result.update(dict(masks=mask_prob))
                
        return result, losses