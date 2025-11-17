import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.double_conv(x)

        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        x = self.down(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, in_height=512, in_width=512, first_stage_channels=32):
        assert in_height % 16 == 0 and in_width % 16 == 0

        super().__init__()

        self.conv_block_1 = ConvBlock(in_channels, first_stage_channels)
        self.down_sampling_1 = DownSampling(first_stage_channels, 2 * first_stage_channels)

        self.conv_block_2 = ConvBlock(2 * first_stage_channels, 2 * first_stage_channels)
        self.down_sampling_2 = DownSampling(2 * first_stage_channels, 4 * first_stage_channels)

        self.conv_block_3 = ConvBlock(4 * first_stage_channels, 4 * first_stage_channels)
        self.down_sampling_3 = DownSampling(4 * first_stage_channels, 8 * first_stage_channels)

        self.conv_block_4 = ConvBlock(8 * first_stage_channels, 8 * first_stage_channels)
        self.down_sampling_4 = DownSampling(8 * first_stage_channels, 16 * first_stage_channels)

        self.conv_block_5 = ConvBlock(16 * first_stage_channels, 16 * first_stage_channels)


    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.down_sampling_1(x1)

        x2 = self.conv_block_2(x2)
        x3 = self.down_sampling_2(x2)

        x3 = self.conv_block_3(x3)
        x4 = self.down_sampling_3(x3)

        x4 = self.conv_block_4(x4)
        bottleneck = self.down_sampling_4(x4)

        bottleneck = self.conv_block_5(bottleneck)

        return x1, x2, x3, x4, bottleneck



if __name__ == "__main__":
    import torch

    img = torch.rand(1, 3, 512, 512)

    model = Encoder(in_channels=3, first_stage_channels=32)

    x1, x2, x3, x4, bottleneck = model(img)

    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(bottleneck.shape)
