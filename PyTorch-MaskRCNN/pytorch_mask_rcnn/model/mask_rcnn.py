from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,  # 输入用于计算特征的backbone网络;分类网络分类类别的数目（加上背景一类）
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # 最小的anchor和GT box之间的IOU，大于它的被认为是正样本;最大的anchor和GT box之间的IOU，小于它的被认为是负样本
                 rpn_num_samples=256, rpn_positive_fraction=0.5,  # 在RPN训练中采样用于计算loss的anchor的数目;RPN训练中正样本所占的比例
                 rpn_reg_weights=(1., 1., 1., 1.),  # 用于编解码bounding boxes
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # 在train中选择bbox中最高的的2000做nms，做完后保留前1000个
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # test同理
                 rpn_nms_thresh=0.7,  # nms的阈值为0.7
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # 最小的实际的bbox与预测的IOU，高于这个被认为是正样本，反之为负样本
                 box_num_samples=512, box_positive_fraction=0.25,  # 训练时候用于计算loss所采样的数量，以及正样本的占比
                 box_reg_weights=(10., 10., 5., 5.),  # 用于编解码bounding boxes
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):  # 只采用cls部分的score大于box_score_thresh的，nms的阈值为0.7，每一类最多检查100个
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels  # 256
        
        # ------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)  # 所采用的anchors的基础大小有这三种
        anchor_ratios = (0.5, 1, 2)  # 所采用的anchors的长宽比有这三种
        num_anchors = len(anchor_sizes) * len(anchor_ratios)  # 总的anchors类型有9种
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)  # 声明RPN结构的网络部分

        # 声明RPN网络的proposal部分
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        # ------------ RoIHeads --------------------------
        # 用于分类和bbox的部分，align到7x7
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2        # 256*7*7
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)  # 定义用于分类和bbox部分的网络
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)

        # 对应maskRCNN的mask预测部分，align到14x14 (因为作者是基于原有的fasterRCNN改的，所以通过以下的方式定义mask部分
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer -------------------------- 将输入的图片缩放到固定的大小以及做归一化
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]  # 记录图片的原始大小
        
        image, target = self.transformer(image, target)  # 将图片做变换
        image_shape = image.shape[-2:]
        feature = self.backbone(image)  # 通过backbone网络得到一个特征，作为RPN的输入和最后三层的输入
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)       # 这个rpn只使用了最后一个featuremap
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)  # 训练下返回两个loss
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)  # 两个全链接网络
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)  # 之后一个输出类别
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)  # 每个类都输出bbox,每一类都有4个，总共num_classes * 4个
        
    def forward(self, x):
        x = x.flatten(start_dim=1)  # 将7x7展平
        x = F.relu(self.fc1(x))  # 共享俩个全链接
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):  # 首先是根据layers，定义了4个3x3的卷积，通道数都为256
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        # 用反卷积网络，stride=2，扩大了特征图到两倍大小
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)  # 用1x1卷积，获得分割结果，binary的,有cls那么多
        super().__init__(d)

        for name, param in self.named_parameters():  # 用kaiming_normal来初始化weight
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
    
class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)       # BN层换成它的原因是 它更适合于小批量，maskrcnn 改成了它The reason why we use FrozenBatchNorm2d instead of BatchNorm2d is that the sizes of the batches are very small, which makes the batch statistics very poor and degrades performance.
        
        for name, parameter in body.named_parameters():
            # print(name)
            # if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:     # 这里改了一下试试
                parameter.requires_grad_(False)         # layer2,3,4 (4,6,3) 需要进行训练,其他层不需要？ 这里之所以没有使用C1，是考虑到由于C1的尺寸过大，训练过程中会消耗很多的显存。 TODO 这里好像C2也没有使用
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)      # 只取前8个，后面的avgpool与fc层去掉
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)      # 这里是变化尺寸以输入到rpn中
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)     # 这里是权重的初始化一种方法，Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差，所以使用了kaiming_uniform_
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):       # TODO 这里整个的是一个顺序网络，没有分支，为何？不是有C2,C3,C4,C5吗？ backbone 多接了1*1和3*3，没有relu
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)  # 变到256， 先1×1
        x = self.layer_block_module(x)  # 再3*3
        return x    # x的尺寸变为1/32

    
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        #backbone_pretrained = True  # 这应该是作者写错了，应该是`pretrained_backbone`
        pretrained_backbone = True
    backbone = ResBackbone('resnet50', pretrained_backbone)  # maskrcnn中用的backbone基于resnet50,并有预训练的模型
    model = MaskRCNN(backbone, num_classes)
    
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])

        # 删除了部分模型中的参数读入
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return model