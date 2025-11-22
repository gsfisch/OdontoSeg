import torch
import torch.nn as nn
from Architecture.MyArchitecture.Encoder import ConvBlock
#from Encoder import ConvBlock

class ConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_decoder, x_encoder):
        diffY = x_encoder.size()[2] - x_decoder.size()[2]
        diffX = x_encoder.size()[3] - x_decoder.size()[3]

        x_decoder = nn.functional.pad(
            x_decoder,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x_encoder, x_decoder], dim=1)

        return x
        

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x):
        x = self.upsampling(x)

        return x


class Decoder(nn.Module):
    def __init__(self, bottleneck_channels=1152, skip_channels=[96, 192, 384, 576], n_classes=4):
        super().__init__()

        self.up_sampling_1 = UpSampling(bottleneck_channels, skip_channels[3])
        self.concat_layer_1 = ConcatLayer()
        self.conv_block_1 = ConvBlock(2 * skip_channels[3], skip_channels[3])


        self.up_sampling_2 = UpSampling(skip_channels[3], skip_channels[2])
        self.concat_layer_2 = ConcatLayer()
        self.conv_block_2 = ConvBlock(2 * skip_channels[2], skip_channels[2])


        self.up_sampling_3 = UpSampling(skip_channels[2], skip_channels[1])
        self.concat_layer_3 = ConcatLayer()
        self.conv_block_3 = ConvBlock(2 * skip_channels[1], skip_channels[1])


        self.up_sampling_4 = UpSampling(skip_channels[1], skip_channels[0])
        self.concat_layer_4 = ConcatLayer()
        self.conv_block_4 = ConvBlock(2 * skip_channels[0], skip_channels[0])


        self.seg_head = nn.Conv2d(skip_channels[0], n_classes, kernel_size=1)

        '''
        self.up_sampling_1 = UpSampling(in_channels, in_channels // 2)
        self.concat_layer_1 = ConcatLayer()
        self.conv_block_1 = ConvBlock(in_channels, in_channels // 2)


        self.up_sampling_2 = UpSampling(in_channels // 2, in_channels // 4)
        self.concat_layer_2 = ConcatLayer()
        self.conv_block_2 = ConvBlock(in_channels // 2, in_channels // 4)


        self.up_sampling_3 = UpSampling(in_channels // 4, in_channels // 8)
        self.concat_layer_3 = ConcatLayer()
        self.conv_block_3 = ConvBlock(in_channels // 4, in_channels // 8)


        self.up_sampling_4 = UpSampling(in_channels // 8, in_channels // 16)
        self.concat_layer_4 = ConcatLayer()
        self.conv_block_4 = ConvBlock(in_channels // 8, in_channels // 16)


        self.seg_head = nn.Conv2d(in_channels // 16, n_classes, kernel_size=1)
        '''




    def forward(self, x1, x2, x3, x4, bottleneck):
        x = self.up_sampling_1(bottleneck)
        x = self.concat_layer_1(x, x4)
        x = self.conv_block_1(x)

        x = self.up_sampling_2(x)
        x = self.concat_layer_2(x, x3)
        x = self.conv_block_2(x)

        x = self.up_sampling_3(x)
        x = self.concat_layer_3(x, x2)
        x = self.conv_block_3(x)

        x = self.up_sampling_4(x)
        x = self.concat_layer_4(x, x1)
        x = self.conv_block_4(x)

        x = self.seg_head(x)

        return x



if __name__ == "__main__":
    import torch

    x1 = torch.rand(1, 96, 128, 128)
    x2 = torch.rand(1, 192, 64, 64)
    x3 = torch.rand(1, 384, 32, 32)
    x4 = torch.rand(1, 576, 16, 16)
    bottleneck = torch.rand(1, 1152, 16, 16)

    model = Decoder(bottleneck_channels=1152, skip_channels=[96, 192, 384, 576])

    logits = model(x1, x2, x3, x4, bottleneck)

    print(logits.shape)
