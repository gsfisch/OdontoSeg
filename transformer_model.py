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
        self.backbone = timm.create_model('swin_base_patch4_window12_384', pretrained=True, features_only=True)
        self.classifier = nn.Conv2d(1024, 4, 1)
        
        
    def forward(self, x):
        # Resize input to backbone resolution
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)

        # Extract features
        features = self.backbone(x)[-1]

        # Pixel wise classification
        output = self.classifier(features.permute(0, 3, 1, 2))

        # Resize output to initial resolution
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output        



arch = SegmentationModel()
