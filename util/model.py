from typing import Optional
import segmentation_models_pytorch as smp
import torch

def make_model(
    encoder: str,
    arch: str,
    classes: Optional[int] = 3) -> torch.nn.Module:
        
    if arch == "U-Net":
        model = smp.Unet(
            encoder_name=encoder,
            classes=classes,
            # activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
        )
    elif arch == "Linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            classes=classes,
            encoder_weights="imagenet",
            encoder_depth= 5,
            decoder_use_batchnorm= True
        )
    elif arch == "FPN":
        model = smp.FPN(
            encoder_name=encoder,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_depth  = 5,
            decoder_pyramid_channels = 256,
            decoder_segmentation_channels = 128,
            decoder_merge_policy = "add",
            decoder_dropout = 0.2,
            upsampling = 4
        )
    elif arch == "PSPNet":
        model = smp.PSPNet(
            encoder_name=encoder,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_depth = 3,
            psp_out_channels = 512,
            psp_use_batchnorm = True,
            psp_dropout = 0.2,
            upsampling = 8,
        )
    elif arch == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=encoder,
            classes=classes,
            encoder_weights="imagenet",
        )
    elif arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            classes=classes,
            encoder_weights="imagenet",
        )

    elif arch == "SegFormer":
        model = smp.SegFormer(
            encoder_name = "resnet34",
            encoder_depth = 5,
            encoder_weights = None, #"imagenet"
            decoder_segmentation_channels = 256,
            in_channels = 3,
            classes = classes,
            activation = None,
            upsampling = 4,
            aux_params = None,
        )
        
    return model