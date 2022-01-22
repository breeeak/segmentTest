import torch


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        # value 是anchors中，所有目标最大的iou，长度等于anchors。matched_idx是对应的哪一个目标，长度等于anchors,最大值是图片中的目标数。最大的iou说明这个框最有可能是这个物体，所以只要这个框即可。
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 
        # 其实是求最有可能有目标的那些框，以及最没有可能的那些框。可能性用的是iou
        label[value >= self.high_threshold] = 1  # 将大于high_threshold的位置定为1
        label[value < self.low_threshold] = 0  # 将小于low_threshold的位置定为0;其余剩下的保持-1
        # 这个策略可不可以没有，会影响到准确率。拿较小的iou来训练势必有影响。但是没有的话可能会报错。matched_idx没有加上这个目标 TODO 尝试改进吧
        if self.allow_low_quality_matches:  # 该部分将最大的IOU值保留作为1,以便当没有IOU大于high_threshold时，仍然有正样本
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]     # 为何不像22行value取值一样方法一样，只要值不要索引？原因是，那样只会返回一个值，这样可以找出所有最大的iou（有可能相等）
            label[gt_pred_pairs] = 1        # 会有重复赋值的可能性。

        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]
        # 有可能比采样数目小 但不会大了， 因为negative 的数量可能小于规定的数量，一般不会
        num_pos = int(self.num_samples * self.positive_fraction)  # 通过采样点数目和positive_fraction计算正样本应有多少
        num_pos = min(positive.numel(), num_pos)  # 比较positive样本的数量和应有正样本数量大小，选取小的那个
        num_neg = self.num_samples - num_pos  # 根据实际采样的正样本数量，计算需要多少负样本
        num_neg = min(negative.numel(), num_neg)  # 比较negative样本的数量和应有负样本数量大小，选取小的那个

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]  # 根据上面计算的数量，随机应提取样本的index
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

    
def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios
        
        self.cell_anchor = None     # 用来记录不同比例大小的形状
        self._cache = {}    # 用于缓存一些计算好的anchor尺寸
        
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return 
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)      # 这里ratio是高比上宽

        h_ratios = torch.sqrt(ratios)  # 算出高上的变化比例  总面积是128*128，所以先开根号，一个边上的比例，再
        w_ratios = 1 / h_ratios  # 算出宽上的变化比例

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)  # 分别算出变化后的9种高   view-1变成1维
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)  # 分别算出变化后的9种宽

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2  # 得到相对应中心点的anchors形状     #为何会有负的？需要的是左上角和右下角4个位置，除以2后，就相当于变到中心了，如果加上anchor中心刚刚好。
        
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]  # 根据两个参数，计算相对左上角的位移， 这里是相对于原图的尺寸，即回到原图
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = torch.meshgrid(shift_y, shift_x)  # 变换成x,y坐标的形式
        x = x.reshape(-1)   # 之所以变成一维的，是不需位置信息了，只要得到总数即可。
        y = y.reshape(-1)
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)  # 变换成4个位置上的shift

        anchor = (shift + self.cell_anchor).reshape(-1, 4)  # 加上之前计算的9中anchor的坐标
        return anchor
        
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:  # 如果这种类型的key组合之前计算过，将直接返回缓存不再计算
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)
        
        if len(self._cache) >= 3:  # 如果缓存的大于等于3个，便清理缓存
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device  # 获得feature的类型和在不在gpu上
        grid_size = tuple(feature.shape[-2:])  # 获得特征的尺寸
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))  # 并计算长宽上所采用的stride为多少
        
        self.set_cell_anchor(dtype, device)  # 计算每个anchors的尺寸
        
        anchor = self.cached_grid_anchor(grid_size, stride)  # 根据这两个参数计算实际的anchors相对左上角的位置
        return anchor