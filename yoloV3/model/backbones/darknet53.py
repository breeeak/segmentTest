import torch.nn as nn
from yoloV3.model.layers.conv_module import Convolutional
from yoloV3.model.layers.blocks_module import Residual_block


class Darknet53(nn.Module):

    def __init__(self):
        """就按照这个定义的顺序来的"""
        super(Darknet53, self).__init__()
        # 输入 416*416*3==>416*416*32
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')
        # 由于stride为2，pad=1, 尺寸会减小  208*208*64   （416+2-3）/2 +1 = 207+1
        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 残差连接 208*208*64
        self.__rb_5_0 = Residual_block(filters_in=64, filters_out=64, filters_medium=32)
        # 由于stride为2，pad=1, 尺寸会减小  104*104*128
        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 2 残差连接
        self.__rb_5_1_0 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        self.__rb_5_1_1 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)

        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 8 残差连接
        self.__rb_5_2_0 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_1 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_2 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_3 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_4 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_5 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_6 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_7 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)

        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 8 残差连接
        self.__rb_5_3_0 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_1 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_2 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_3 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_4 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_5 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_6 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_7 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)

        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 4 残差连接
        self.__rb_5_4_0 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_1 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_2 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_3 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)


    def forward(self, x):
        x = self.__conv(x)

        x0_0 = self.__conv_5_0(x)
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1)
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2)
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)  # small （52*52*256）

        x3_0 = self.__conv_5_3(x2_8)
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)  # medium （26*26*512）

        x4_0 = self.__conv_5_4(x3_8)
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)  # large （13*13*1024）

        return x2_8, x3_8, x4_4

if __name__ == '__main__':
    # 测试模型是否可行
    import torch
    model = Darknet53()
    model.eval()
    image = torch.randn(1, 3, 416, 416)
    output = model(image)

