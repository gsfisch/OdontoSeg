import math
import torch
import torch.nn as nn


# Encoder
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1 conv to match dimensions for residual path
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.residual(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = ResidualConv(3, 64, stride=1)
        self.down1 = ResidualConv(64, 128, stride=2)
        self.down2 = ResidualConv(128, 256, stride=2)
        self.down3 = ResidualConv(256, 512, stride=2)
        self.down4 = ResidualConv(512, 1024, stride=2)

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        return x1, x2, x3, x4, x5


# Transformer
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D sinusoidal encoding.")
        self.channels = channels

    def forward(self, H, W, device):
        """
        Returns a positional encoding tensor of shape (H*W, C)
        """

        # Create grid of positions
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        # y, x shapes: (H, W)

        pe = torch.zeros(self.channels, H, W, device=device)

        # Divide channels for sin/cos of x/y
        c = self.channels // 4

        # Calculate frequency range
        div_term = torch.exp(
            torch.arange(0, c, 1, device=device) * (-math.log(10000.0) / c)
        )

        # Apply sin & cos
        pe[0:c, :, :]      = torch.sin(x.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[c:2*c, :, :]    = torch.cos(x.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[2*c:3*c, :, :]  = torch.sin(y.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[3*c:4*c, :, :]  = torch.cos(y.unsqueeze(0) * div_term.view(-1, 1, 1))

        # Return (H*W, C)
        return pe.reshape(self.channels, H * W).permute(1, 0)  # (HW, C)


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=4, num_layers=2, dim_feedforward=1024):
        super().__init__()

        self.channels = channels
        self.pos_encoding = PositionalEncoding2D(channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Flatten: (B, C, H*W) → (B, H*W, C)
        seq = x.flatten(2).permute(0, 2, 1)

        # Positional Encoding: (H*W, C)
        pos = self.pos_encoding(H, W, device)

        # Broadcast to batch: (B, H*W, C)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        # Add positional encoding
        seq = seq + pos

        # Transformer
        seq = self.transformer(seq)

        # Back to feature map shape
        return seq.permute(0, 2, 1).view(B, C, H, W)


# Decoder
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = ResidualConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Fix potential shape mismatch
        if x.size() != skip.size():
            x = F.pad(x, [0, skip.size(3)-x.size(3),
                          0, skip.size(2)-x.size(2)])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)



# Complete Architecture
class UNetWithTransformer(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.encoder = Encoder()
        self.transformer = TransformerBottleneck(1024)
        self.decoder = Decoder()

        # Override number of output classes
        self.decoder.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.transformer(x5)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out



