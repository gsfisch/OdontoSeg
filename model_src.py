import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            # Padding or cropping if needed
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        #return self.conv(x)

        # Dropout
        x = nn.Dropout2d(p=0.2)(x)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[::-1]  # from deepest to shallowest
        self.center = ConvBlock(encoder_channels[0], decoder_channels[0])

        self.blocks = nn.ModuleList([
            DecoderBlock(decoder_channels[i], encoder_channels[i + 1], decoder_channels[i + 1])
            for i in range(len(decoder_channels) - 1)
        ])

        self.last_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), # 128
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), # 256
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)   # 512
        )

    def forward(self, features):
        # features: list of feature maps from encoder, deepest to shallowest
        x = self.center(features[-1].permute(0, 3, 1, 2))  # start from deepest
        for i, decoder_block in enumerate(self.blocks):
            skip = features[-(i + 2)].permute(0, 3, 1, 2)  # skip connections from encoder
            x = decoder_block(x, skip)

        x = self.last_blocks(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = timm.create_model(
            'swinv2_small_window8_256',
            pretrained=True,
            features_only=True
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        #for name, param in self.encoder.named_parameters():
        #    #print(f'Name: {name}')
        #    if 'layers_3.blocks.1' in name or 'layers_3.blocks.0' in name:
        #        param.requires_grad = True
        #    else:
        #        param.requires_grad = False

        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = [512, 256, 128, 64]  # Customize as needed

        self.decoder = UNetDecoder(encoder_channels, decoder_channels)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)
        #self.final_conv = torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        features = self.encoder(x)  # List of feature maps

        x = self.decoder(features)
        #print(f'Depois do decoder: {x.shape}')

        x = self.final_conv(x)
        #print(f'Depois do classifier: {x.shape}')
        return x
