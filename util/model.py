from typing import Optional
import segmentation_models_pytorch as smp
import torch
import torchseg
import torch.nn as nn
import torch.nn.functional as F
from Architecture.MyArchitecture.MyArchitecture import MyArchitecture
from Architecture.MyArchitecture_v2.MyArchitecture_v2 import MyArchitecture_v2
from Architecture.MyArchitecture_pvt_v2_b1__U_Net.MyArchitecture_pvt_v2_b1__U_Net import MyArchitecture_pvt_v2_b1__U_Net



class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, x):
        y = self.model(x)
        y = F.interpolate(y, size=(512, 512), mode='bilinear', align_corners=False)

        return y



def make_model(
    encoder: str,
    arch: str,
    classes: Optional[int] = 3,
    library='smp',
    encoder_depth=0,
    decoder_channels=(),
    encoder_params={},
    head_upsampling=2,
    ) -> torch.nn.Module:

    used_wrapper = False

    if arch == 'MyArchitecture':
        model = MyArchitecture()

        for param in model.transformer.parameters():
            param.requires_grad = False
        
        return model

    if arch == 'MyArchitecture_v2':
        model = MyArchitecture_v2()

        for param in model.transformer.parameters():
            param.requires_grad = False
        
        return model

    if arch == 'MyArchitecture_pvt_v2_b1__U_Net':
        model = MyArchitecture_pvt_v2_b1__U_Net()

        for param in model.transformer.parameters():
            param.requires_grad = False
        
        return model



    if library == 'smp': 
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
    elif library == 'torchseg':
        if arch == 'U-Net':
            model = torchseg.Unet(
            encoder_name=encoder,
            encoder_weights=True,
            in_channels=3,
            classes=classes,
            decoder_channels=decoder_channels,
            encoder_depth=encoder_depth,
            encoder_params=encoder_params,
            head_upsampling=head_upsampling,
            )

        elif arch == 'U-Net++':
            model = torchseg.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                decoder_channels=decoder_channels,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

            model = Wrapper(model)
            used_wrapper = True

        elif arch == 'MAnet':
            model = torchseg.MAnet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                decoder_channels=decoder_channels,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

            model = Wrapper(model)
            used_wrapper = True

        elif arch == 'Linknet':
            model = torchseg.Linknet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'FPN':
            model = torchseg.FPN(
            encoder_name=encoder,
            encoder_weights=True,
            in_channels=3,
            classes=classes,
            encoder_params=encoder_params,
            )

        elif arch == 'PSPNet':
            model= torchseg.PSPNet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'PAN':
            model = torchseg.PAN(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'DeepLabV3':
            model = torchseg.DeepLabV3(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'DeepLabV3+':
            model = torchseg.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        else:
            print("Architecture not implemented")

    
    # Freeze encoder
    if used_wrapper:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
        
    return model


def make_custom_model():
    pass