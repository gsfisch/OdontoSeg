import numpy as np
from typing import Optional
import segmentation_models_pytorch as smp
import torch
import torchseg
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR, SwinUNETR
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
#from Architecture.MyArchitecture.MyArchitecture import MyArchitecture
#from Architecture.MyArchitecture_v2.MyArchitecture_v2 import MyArchitecture_v2
#from Architecture.MyArchitecture_pvt_v2_b1__U_Net.MyArchitecture_pvt_v2_b1__U_Net import MyArchitecture_pvt_v2_b1__U_Net
#from Architecture.MyArchitecture_with_skip_connections.MyArchitecture_with_skip_connections import MyArchitecture_with_skip_connections
#from transformers import EoMTFoSemanticSegmentation, EoMTFeatureExtractor
#from transformers import EomtForUniversalSegmentation, EomtImageProcessor
#from transformers import AutoConfig, AutoModelForSemanticSegmentation
#from transformers import EomtConfig, EomtForUniversalSegmentation
#from timm.models.swin_transformer import swin_tiny_patch4_window7_224 


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
    freeze_encoder = False,
    head_upsampling=2,
    need_wrapper=False,
    ) -> torch.nn.Module:

    used_wrapper = False


    if library == '':
        if arch == 'TransUNet':
            print('TransUNet')

            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = 4
            config_vit.n_skip = 3
            config_vit.pretrained_path = '/home/fisch/Documents/OdontoSeg/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
            config_vit.patches.grid = (32, 32)


            model = ViT_seg(config_vit, img_size=512, num_classes=classes).cuda()
            model.load_from(weights=np.load(config_vit.pretrained_path))


            return model


    if library == 'monai':
        if arch == 'SwinUNETR':
            print("SwinUNETR")
            model = SwinUNETR(
                in_channels=3,
                out_channels=4,
                spatial_dims=2,
                feature_size=36,
                use_checkpoint=True
            )

            return model

        elif arch == 'UNETR':
            
            model = UNETR(
                in_channels=3,
                out_channels=4,
                spatial_dims=2,
                img_size=(512, 512),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                #pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.1,
            )
            

            return model

    '''
    if arch == 'EoMT':
        #config = EomtConfig(image_size=[512, 512])
        #from transformers import EomtConfig

        config = EomtConfig(
            image_size=512,
            patch_size=16,
            num_channels=3,
            num_labels=4,

            # Transformer backbone
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,

            # Dropout / regularization
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,

            # Labels
            id2label={
                0: "class_0",
                1: "class_1",
                2: "class_2",
                3: "class_3",
            },
            label2id={
                "class_0": 0,
                "class_1": 1,
                "class_2": 2,
                "class_3": 3,
            },
        )

        model = EomtForUniversalSegmentation(config)

        return model
    ''

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

    if arch == 'MyArchitecture_with_skip_connections':
        model = MyArchitecture_with_skip_connections()

        for param in model.transformer.parameters():
            param.requires_grad = False
        
        return model
    '''

    if library == 'smp':
        if arch == 'SegFormer':
            print("SegFormer")

            model = smp.Segformer(
                encoder_name=encoder,
                in_channels=3,
                classes=classes,
            )

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
                #activation="softmax" if classes > 1 else "sigmoid",
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

            if need_wrapper:
                model = Wrapper(model)


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

            if need_wrapper:
                model = Wrapper(model)
            


        elif arch == 'Linknet':
            model = torchseg.Linknet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

            if need_wrapper:
                model = Wrapper(model)


        elif arch == 'FPN':
            model = torchseg.FPN(
            encoder_name=encoder,
            encoder_weights=True,
            in_channels=3,
            classes=classes,
            encoder_depth=encoder_depth,
            encoder_params=encoder_params,
            )

            if need_wrapper:
                model = Wrapper(model)


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
            exit()
        
        if freeze_encoder:
            if need_wrapper:
                for param in model.model.encoder.parameters():
                    param.requires_grad = False

            else:
                for param in model.encoder.parameters():
                    param.requires_grad = False


    return model


def make_custom_model():
    pass