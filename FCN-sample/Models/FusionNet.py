import torch.nn as nn
import torch


def conv_block(in_dim,out_dim,act_fn,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_3(in_dim, out_dim, act_fn):
    """
    这里输出没有act_fn,残差块
    :param in_dim:
    :param out_dim:
    :param act_fn:
    :return:
    """
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


class Conv_residual_conv(nn.Module):    # 汉堡块
    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)     # 绿色块
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)  # 紫色块
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)    # 绿色块

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2   # 跳跃连接，是第一个绿色块输出加紫色块输出
        conv_3 = self.conv_3(res)

        return conv_3


class Fusionnet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, out_clamp=None):
        super(Fusionnet, self).__init__()
        # 3, 12, 64
        self.in_dim = input_nc
        self.final_out_dim = output_nc
        self.out_dim = ngf
        self.out_clamp = out_clamp

        act_fn = nn.ReLU()
        act_fn_2 = nn.ELU(inplace=True)

        # encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)     # 汉堡块
        # 这里用步长为2的卷积来替代最大池化,达到下采样的效果。
        self.pool_1 = conv_block(self.out_dim, self.out_dim, act_fn, 2)     # 蓝色块
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)    # ×2的变化
        self.pool_2 = conv_block(self.out_dim * 2, self.out_dim * 2, act_fn, 2)
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = conv_block(self.out_dim * 4, self.out_dim * 4, act_fn, 2)
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = conv_block(self.out_dim * 8, self.out_dim * 8, act_fn, 2)

        # bridge
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder
        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)     # 红色块
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output  这里是单独定义的，其实更为常用的是kernel=1,没有batch_normal,没有激活，输出通道对应类别
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2    # 前面的特征融合（跳跃连接），这里是直接相加，取平均值
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)

        return out


if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 352, 480)

    net = Fusionnet(3, 12, 64)

    out = net(rgb)

    print(out.shape)
