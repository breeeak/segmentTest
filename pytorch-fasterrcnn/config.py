import numpy as np

class_to_ind = {
    'BG':0,           'aeroplane':1,       'bicycle':2,    'bird':3,
    'boat':4,    'bottle':5, 'bus':6,    'car':7,
    'cat':8, 'chair':9,     'cow':10,       'diningtable':11,
    'dog':12,   'horse':13,     'motorbike':14,    'pottedplant':15,
    'sheep':16,'sofa':17,   'train':18,    'tvmonitor':19,
    'person':20
}
# 这里只使用voc2012进行训练了，要改的话，修改data/pascalvoc的处理方式即可
trainset_root_path = 'E:/1_dataset/common_dataset/VOC/data/VOC_directly/VOC2012/'
testset_root_path = 'E:/1_dataset/common_dataset/VOC/data/VOC_directly/VOC2012/'
rng_seed = 3
batch_size = 1  #目前只支持batch_size = 1!
epoch = 20
learning_rate = 0.001
lr_decay_step = 5  #epoch
lr_decay = 0.1
checkpoint_interval = 1     # 多久保存一次
checkpoint_name = 'fasterrcnn_r50'
pre_train = True
clip_grad = True
work_dir = './work_dir'
load_from = None        #'./work_dir/fasterrcnn_r50-19.pth' 预训练的路径
truncated = False

image_max_size = 1000
image_min_size = 600
#300*500 600*1000 s=2
#300*400 600*800  s=2
#600*400 900*600  s=1.5
pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])   #BGR order，用于图像归一化的

backbone='resnet50'
rpn_featurechannels = 512       # featuremaps 512

anchor_scales = [8, 16, 32]     # 不同大小
anchor_ratios = [0.5, 1.0, 2.0]     #不同的长宽比例

anchor_batch_size = 256     # 训练rpn的时候anchor的batchsize
rpn_pos_iou_thr = 0.7       # 征服样本的iou阈值
rpn_neg_iou_thr = 0.3

rpn_nms_thresh = 0.7        # nms的阈值

train_rpn_pre_nms_top_N = 12000     # 保留前多少个rpn,nms之前之后的
train_rpn_post_nms_top_N = 2000
test_rpn_pre_nms_top_N = 6000
test_rpn_post_nms_top_N = 300

rpn_bbox_inside_weight = (1.0, 1.0, 1.0, 1.0)   # 权重 不需要改动

proposal_batch_size = 128   # proposal层的batchsize
proposal_fg_fraction = 0.25
proposal_fg_thresh = 0.5
proposal_bg_thresh_hi = 0.5
proposal_bg_thresh_lo = 0.1

bbox_normalize_targets_precomputed = True
bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
bbox_normalize_std = (0.1, 0.1, 0.2, 0.2)
bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)

pooling_mode = 'align'
pool_size = 7

rcnn_feat_channels = 2048

test_nms_threshold = 0.3


