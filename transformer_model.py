import timm
import torch.nn as nn
import torch.nn.functional as F


#print(timm.list_models()[:])


# 'swin_base_patch4_window7_224', 'swin_base_patch4_window12_384', 'swin_large_patch4_window7_224', 'swin_large_patch4_window12_384', 'swin_s3_base_224', 'swin_s3_small_224',
# 'swin_s3_tiny_224', 'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224', 'swinv2_base_window8_256',
# 'swinv2_base_window12_192', 'swinv2_base_window12to16_192to256', 'swinv2_base_window12to24_192to384', 'swinv2_base_window16_256',
# 'swinv2_cr_base_224', 'swinv2_cr_base_384', 'swinv2_cr_base_ns_224', 'swinv2_cr_giant_224', 'swinv2_cr_giant_384', 'swinv2_cr_huge_224',
# 'swinv2_cr_huge_384', 'swinv2_cr_large_224', 'swinv2_cr_large_384', 'swinv2_cr_small_224', 'swinv2_cr_small_384', 'swinv2_cr_small_ns_224',
# 'swinv2_cr_small_ns_256', 'swinv2_cr_tiny_224', 'swinv2_cr_tiny_384', 'swinv2_cr_tiny_ns_224', 'swinv2_large_window12_192',
# 'swinv2_large_window12to16_192to256', 'swinv2_large_window12to24_192to384', 'swinv2_small_window8_256', 'swinv2_small_window16_256', 'swinv2_tiny_window8_256', 'swinv2_tiny_window16_256'


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('swinv2_small_window8_256', pretrained=True, features_only=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2), # 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # 512
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(16, 4, 1)
        
        
    def forward(self, x):
        # Resize input to encoder resolution
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # Extract features
        features = self.encoder(x)[-1]

        # Decode features
        decoded_features = self.decoder(features.permute(0, 3, 1, 2))

        # Pixel wise classification
        output = self.classifier(decoded_features)

        # Resize output to initial resolution
        #output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output        
