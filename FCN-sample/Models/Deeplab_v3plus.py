import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    """
    自定义padding，     自定义padding，
    这里是为了防止卷积核尺寸为偶数时，导致尺寸计算失误。pytorch中可以为偶数，但是会进行取整等运算
    :param inputs:
    :param kernel_size:
    :param dilation:
    :return:
    """
    # 空洞卷积所对应的实际卷积尺寸
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    """
    深度可分离卷积
    """
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()
        # group 参数的意思就是要分多少个组，就是几个通道对应几个卷积，深度卷积就是一个通道的特征对应一个卷积，所以groups=inplanes, 最后也会转化成定义的输出的尺寸
        # 深度卷积
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        # 逐点卷积
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    """
    stride 表示最大的步长
    reps 表示有几个一模一样的卷积层
    start_with_relu  block是否从reLU 开始，即是否是先需要激活，第一个肯定不用
    is_last 是否是最后一个模块
    """
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        # 这里是跳跃连接，对应图中的红色部分，就是改编的那里
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            # 否则就没有，一路平坦
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        # 往block 里扔， 这里才是真正的结构
        rep = []
        filters = inplanes
        if grow_first:
            # 定义每个block 里的普通的卷积
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
            # 这里是为了设置后面的通道数
            filters = planes
        # 之所以要这样是因为通道数发生改变，block的后面几层与前面第一个是不一样的
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]
        # 只有红色 stride=2的那层
        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))
        # 这里是最后的那层，之所以做了一下stride的判断，是因为os=8的时候，前面几个block的stride是1
        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))
        # 最后将数组解成顺序模型
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            # 之所以这样，是因为middle flow没有经过卷积,而是直接与前面的相加
            skip = inp
        # 这里加 就相当于融合
        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)

        # Exit flow
        # 这个block 对原图进行了改编，stride设为了1
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # 之所以要保留一个，是因为DCNN,即Xception中的相同维度的一个层给拿出来，与后面的上采样合并
        # 拿出来一个低尺度特征 通道128,尺寸是原图的1/4)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, os):
        super(ASPP_module, self).__init__()
        # ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        # ASPP 是并行的，同一个输入，不同的输出
        # 空洞卷积，要想尺寸不变，需要padding=dilation,对于3×3卷积而言
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                                                          padding=0, dilation=dilations[0], bias=False),
                                                nn.BatchNorm2d(planes),
                                                nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                                           padding=dilations[1], dilation=dilations[1], bias=False),
                                                nn.BatchNorm2d(planes),
                                                nn.ReLU())
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                                           padding=dilations[2], dilation=dilations[2], bias=False),
                                                nn.BatchNorm2d(planes),
                                                nn.ReLU())
        self.aspp4 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                                           padding=dilations[3], dilation=dilations[3], bias=False),
                                                 nn.BatchNorm2d(planes),
                                                 nn.ReLU())
        # 这里无论输入都抻开成了一维向量，可以理解成一个特征取了一个平均值，通道数不变，所以还是inplanes
        #    不管之前的特征图尺寸为多少，只要设置为(1,1)，那么最终特征图大小都为(1,1)，通道数不变
        #     The number of output features is equal to the number of input planes.
        # 再做1×1卷积，之所以256，是需要跟上面的做concat,通道数要一样
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        # 1×1卷积，concat后，通道数叠加，所以是5*256, 这里在后面单独用了
        # self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(256)
        # 文章中的初始化权重方法
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # X5 由于HW尺寸不一样是不能concat的，需要上采样，这里采用双线性插值，比较粗暴，记住就行比较常用，对于语义分割而言，align=true
        # 根据指定的尺寸进行上采样
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os)

        self.ASPP = ASPP_module(2048, 256, 16)
        # 这里1280是拼接后图层的大小
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x = self.ASPP(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 这里上采样到原图的1/4大小，  后面只剩一个上采样到原图大小
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
        # 这里前面经过了两个stride=2，所以也是1/4大小，可以用于后面拼接，128--> 48
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        # low_level 通道数变成了48，x是256，拼接后是304
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        # 这里最后上采样到和原图一样大
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 测试模型是否可行
    model = DeepLabv3_plus(nInputChannels=3, n_classes=12, os=16, _print=True)
    model.eval()
    image = torch.randn(1, 3, 352, 480)
    output = model(image)
    print(output.size())

    # # 测试ASPP, 期望88通道输入，256来输出
    # model = ASPP_module(88, 256, 16)
    # model.eval()
    # image = torch.randn(1, 88, 352, 480)       # batch, channel, H, W
    # output = model(image)
    # print(output.size())