from typing import Optional
import segmentation_models_pytorch as smp
import torch
import torchseg
import torch.nn as nn
import torch.nn.functional as F
from Architecture.MyArchitecture.MyArchitecture import MyArchitecture
from Architecture.MyArchitecture_v2.MyArchitecture_v2 import MyArchitecture_v2
from Architecture.MyArchitecture_pvt_v2_b1__U_Net.MyArchitecture_pvt_v2_b1__U_Net import MyArchitecture_pvt_v2_b1__U_Net
from Architecture.MyArchitecture_with_skip_connections.MyArchitecture_with_skip_connections import MyArchitecture_with_skip_connections
#from transformers import EoMTFoSemanticSegmentation, EoMTFeatureExtractor
from transformers import EomtForUniversalSegmentation, EomtImageProcessor
#from transformers import AutoConfig, AutoModelForSemanticSegmentation
from transformers import EomtConfig, EomtForUniversalSegmentation
from monai.networks.nets import UNETR, SwinUNETR
from timm.models.swin_transformer import swin_tiny_patch4_window7_224


class SwinUNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, pretrained=True):
        super().__init__()

        self.encoder = swin_tiny_patch4_window7_224(pretrained=pretrained)

        if in_channels != 3:
            self.encoder.patch_embed.proj = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=4
            )

        # Decoder
        self.up1 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(96, 96, 2, stride=2)

        self.conv1 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv2 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)

        self.out = nn.Conv2d(96, num_classes, 1)

    def forward(self, x):
        # Interpolation
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Encoder
        x = self.encoder.patch_embed(x)

        features = []
        for layer in self.encoder.layers:
            x = layer(x)
            features.append(x)

        # Extract multi-scale features
        x1 = features[0].permute(0, 3, 1, 2)
        x2 = features[1].permute(0, 3, 1, 2)
        x3 = features[2].permute(0, 3, 1, 2)
        x4 = features[3].permute(0, 3, 1, 2)
        

        # Decoder
        d4 = self.up1(x4)
        d4 = self.conv1(d4 + x3)

        d3 = self.up2(d4)
        d3 = self.conv2(d3 + x2)

        d2 = self.up3(d3)
        d2 = self.conv3(d2 + x1)

        d1 = self.up4(d2)

        y = F.interpolate(self.out(d1), size=(512, 512), mode='bilinear', align_corners=False)

        return y



class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, x):
        y = self.model(x)
        y = F.interpolate(y, size=(512, 512), mode='bilinear', align_corners=False)

        return y


class eomt_wrapper(nn.Module):
    def __init__(self, eomt, processor):
        super().__init__()

        self.eomt = eomt
        self.processor = processor

    def forward(self, x):
        encoder_output = self.eomt(x)

        output = self.processor.post_process_semantic_segmentation(encoder_output, target_sizes=[512, 512, 4])

        return output


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
 

    if library == 'smp':
        print("Using library: SMP")

        if arch == 'SegFormer':
            print("Using architecture: SegFormer")

            model = smp.Segformer(
                encoder_name=encoder,
                in_channels=3,
                classes=classes,
            )

        elif arch == "U-Net":
            print("Using architecture: U-Net")

            model = smp.Unet(
                encoder_name=encoder,
                classes=classes,
                # activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
            )

        elif arch == "Linknet":
            print("Using architecture: Linknet")

            model = smp.Linknet(
                encoder_name=encoder,
                classes=classes,
                encoder_weights="imagenet",
                encoder_depth= 5,
                decoder_use_batchnorm= True
            )

        elif arch == "FPN":
            print("Using architecture: FPN")

            model = smp.FPN(
                encoder_name=encoder,
                classes=classes,
                activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
                encoder_depth  = encoder_depth,
                decoder_pyramid_channels = 256,
                decoder_segmentation_channels = 128,
                decoder_merge_policy = "add",
                decoder_dropout = 0.2,
                upsampling = head_upsampling
            )

        elif arch == "PSPNet":
            print("Using architecture: PSPNet")

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
            print("Using architecture: DeepLabV3")

            model = smp.DeepLabV3(
                encoder_name=encoder,
                classes=classes,
                encoder_weights="imagenet",
            )

        elif arch == "DeepLabV3Plus":
            print("Using architecture: DeepLabV3Plus")

            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                classes=classes,
                encoder_weights="imagenet",
            )

    elif library == 'torchseg':
        print("Using library: TorchSeg")

        if arch == 'U-Net':
            print("Using architecture: U-Net")

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
            print("Using architecture: U-Net++")

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
            print("Using architecture: MAnet")

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
            print("Using architecture: Linknet")

            model = torchseg.Linknet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

            model = Wrapper(model)
            used_wrapper = True

        elif arch == 'FPN':
            print("Using architecture: FPN")

            model = torchseg.FPN(
            encoder_name=encoder,
            encoder_weights=True,
            in_channels=3,
            classes=classes,
            encoder_params=encoder_params,
            encoder_depth=5
            )

        elif arch == 'PSPNet':
            print("Using architecture: PSPNet")

            model= torchseg.PSPNet(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'PAN':
            print("Using architecture: PAN")

            model = torchseg.PAN(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'DeepLabV3':
            print("Using architecture: DeepLabV3")

            model = torchseg.DeepLabV3(
                encoder_name=encoder,
                encoder_weights=True,
                in_channels=3,
                classes=classes,
                encoder_depth=encoder_depth,
                encoder_params=encoder_params,
            )

        elif arch == 'DeepLabV3+':
            print("Using architecture: DeepLabV3+")

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

    elif library == 'monai':
        print('Using library: MONAI')

        if arch == 'SwinUNETR':
            print("Using architecture: SwinUNETR")

            model = SwinUNETR(
                #img_size=(512, 512),
                in_channels=3,
                out_channels=4,
                spatial_dims=2,
                feature_size=36,
                use_checkpoint=True
            )
            '''
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
            '''

            return model

    elif library == '':
        print("Using library: None")

        if arch == 'Swin-UNet':
            print("Using architecture: Swin-UNet")

            model = SwinUNet()

            return model

        elif arch == 'MyArchitecture':
            print(f"Using architecture: {arch}")

            model = MyArchitecture()

            for param in model.transformer.parameters():
                param.requires_grad = False
            
            return model

        elif arch == 'MyArchitecture_v2':
            print(f"Using architecture: {arch}")

            model = MyArchitecture_v2()

            for param in model.transformer.parameters():
                param.requires_grad = False
            
            return model

        elif arch == 'MyArchitecture_pvt_v2_b1__U_Net':
            print(f"Using architecture: {arch}")

            model = MyArchitecture_pvt_v2_b1__U_Net()

            for param in model.transformer.parameters():
                param.requires_grad = False
            
            return model

        elif arch == 'MyArchitecture_with_skip_connections':
            print(f"Using architecture: {arch}")

            model = MyArchitecture_with_skip_connections()

            for param in model.transformer.parameters():
                param.requires_grad = False
            
            return model

    
    # Freeze encoder
    if used_wrapper:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
        
    return model



'''
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

    if library == '':

        if arch == 'Swin-UNet':
            model = SwinUNet()

            return model

    if library == 'monai':
        if arch == 'SwinUNETR':
            print("SwinUNETR")
            model = SwinUNETR(
                #img_size=(512, 512),
                in_channels=3,
                out_channels=4,
                spatial_dims=2,
                feature_size=36,
                use_checkpoint=True
            )
            ''
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
            ''

            return model

    ''
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
'''

def make_custom_model():
    pass