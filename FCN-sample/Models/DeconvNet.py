import torch
import torchvision.models as models
from torch import nn


vgg16_pretrained = models.vgg16(pretrained=False)


def decoder(input_channel, output_channel, num=3):
    # 这里deconvnet用的是反卷积，但是步长是1，所以尺寸变化和卷积是一样的
    if num == 3:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))
    elif num == 2:
        decoder_body = nn.Sequential(
            nn.ConvTranspose2d(input_channel, input_channel, 3, padding=1),
            nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1))

    return decoder_body


class VGG16_deconv(torch.nn.Module):
    def __init__(self):
        super(VGG16_deconv, self).__init__()

        pool_list = [4, 9, 16, 23, 30]
        for index in pool_list:
            vgg16_pretrained.features[index].return_indices = True

        self.encoder1 = vgg16_pretrained.features[:4]
        self.pool1 = vgg16_pretrained.features[4]

        self.encoder2 = vgg16_pretrained.features[5:9]
        self.pool2 = vgg16_pretrained.features[9]

        self.encoder3 = vgg16_pretrained.features[10:16]
        self.pool3 = vgg16_pretrained.features[16]

        self.encoder4 = vgg16_pretrained.features[17:23]
        self.pool4 = vgg16_pretrained.features[23]

        self.encoder5 = vgg16_pretrained.features[24:30]
        self.pool5 = vgg16_pretrained.features[30]

        self.classifier = nn.Sequential(
            # 这里必须对应 输入的尺寸 变为1/32, batchsize 设为了1,所以是这样 要是有batchsize还要再乘
            torch.nn.Linear(512 * 11 * 15, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 512 * 11 * 15),   # 再变回去
            torch.nn.ReLU(),
        )

        self.decoder5 = decoder(512, 512)
        self.unpool5 = nn.MaxUnpool2d(2, 2)

        self.decoder4 = decoder(512, 256)
        self.unpool4 = nn.MaxUnpool2d(2, 2)

        self.decoder3 = decoder(256, 128)
        self.unpool3 = nn.MaxUnpool2d(2, 2)

        self.decoder2 = decoder(128, 64, 2)
        self.unpool2 = nn.MaxUnpool2d(2, 2)

        self.decoder1 = decoder(64, 12, 2)
        self.unpool1 = nn.MaxUnpool2d(2, 2)

    def forward(self, x):                       # 3, 352, 480
        encoder1 = self.encoder1(x)             # 64, 352, 480
        output_size1 = encoder1.size()          # 64, 352, 480
        pool1, indices1 = self.pool1(encoder1)  # 64, 176, 240

        encoder2 = self.encoder2(pool1)         # 128, 176, 240
        output_size2 = encoder2.size()          # 128, 176, 240
        pool2, indices2 = self.pool2(encoder2)  # 128, 88, 120

        encoder3 = self.encoder3(pool2)         # 256, 88, 120
        output_size3 = encoder3.size()          # 256, 88, 120
        pool3, indices3 = self.pool3(encoder3)  # 256, 44, 60

        encoder4 = self.encoder4(pool3)         # 512, 44, 60
        output_size4 = encoder4.size()          # 512, 44, 60
        pool4, indices4 = self.pool4(encoder4)  # 512, 22, 30

        encoder5 = self.encoder5(pool4)         # 512, 22, 30
        output_size5 = encoder5.size()          # 512, 22, 30
        pool5, indices5 = self.pool5(encoder5)  # 512, 11, 15

        pool5 = pool5.view(pool5.size(0), -1)   # 变回一维的尺寸
        fc = self.classifier(pool5)
        fc = fc.reshape(1, 512, 11, 15)     # 再变回二维的尺寸。
        # 因此当不确能否使用view时，可以使用reshape。如果只是想简单地重塑一个tensor的shape，那么就是用reshape，
        # 但是如果需要考虑内存的开销而且要确保重塑后的tensor与之前的tensor共享存储空间，那就使用view()。
        # 可以简单认为深浅拷贝
        unpool5 = self.unpool5(input=fc, indices=indices5, output_size=output_size5)    # 512, 22, 30
        decoder5 = self.decoder5(unpool5)   # 512, 22, 30

        unpool4 = self.unpool4(input=decoder5, indices=indices4, output_size=output_size4)  # 512, 44, 60
        decoder4 = self.decoder4(unpool4)   # 256, 44, 60

        unpool3 = self.unpool3(input=decoder4, indices=indices3, output_size=output_size3)  # 256, 88, 120
        decoder3 = self.decoder3(unpool3)   # 128, 88, 120

        unpool2 = self.unpool2(input=decoder3, indices=indices2, output_size=output_size2)  # 128, 176, 240
        decoder2 = self.decoder2(unpool2)  # 64, 176, 240

        unpool1 = self.unpool1(input=decoder2, indices=indices1, output_size=output_size1)  # 64, 352, 480
        decoder1 = self.decoder1(unpool1)  # 12, 352, 480

        return decoder1


if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 352, 480)

    net = VGG16_deconv()

    out = net(rgb)

    print(out.shape)