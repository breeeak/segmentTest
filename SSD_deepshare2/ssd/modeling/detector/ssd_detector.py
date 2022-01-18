from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)

        #设置.train()，self.training=True 训练阶段模型返回为loss 包括reg_loss和cls_logg
        #设置.eval()，self.training=False 验证阶段模型返回为检测结果
        if self.training:
            return detector_losses
        return detections
