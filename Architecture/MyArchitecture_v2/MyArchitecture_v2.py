import torch.nn as nn
import torch.nn.functional as F
from Architecture.MyArchitecture.Encoder import Encoder
from Architecture.MyArchitecture.Decoder import Decoder
from Architecture.MyArchitecture.Transformer import Transformer
#from Encoder import Encoder
#from Decoder import Decoder
#from Transformer import Transformer




class MyArchitecture_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(in_channels=3, encoder_channels=[96, 192, 384, 576, 1152])
        self.transformer = Transformer()
        self.decoder = Decoder(bottleneck_channels=1152, skip_channels=[96, 192, 384, 576])


    def forward(self, x):
        # Encoder
        _, _, _, _, bottleneck = self.encoder(x)


        # Transformer
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)      
        feature_maps_list = self.transformer(x)
        #x1 = feature_maps_list[0].permute(0, 3, 1, 2).contiguous()
        #x2 = feature_maps_list[1].permute(0, 3, 1, 2).contiguous()
        #x3 = feature_maps_list[2].permute(0, 3, 1, 2).contiguous()
        #x4 = feature_maps_list[3].permute(0, 3, 1, 2).contiguous()

        #print(f'x shape: {x.shape}')
        #print(f'x1 shape: {x1.shape}')
        #print(f'x2 shape: {x2.shape}')
        #print(f'x3 shape: {x3.shape}')
        #print(f'x4 shape: {x4.shape}')
        #print(f'bottleneck shape: {bottleneck.shape}')


        # Decoder
        logits = self.decoder(feature_maps_list[0], feature_maps_list[1], feature_maps_list[2], feature_maps_list[3], bottleneck)

        #print(f'Logits shape: {logits.shape}')
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)  


        return logits


'''
class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(in_channels=3, first_stage_channels=32)
        self.transformer = Transformer()
        self.decoder = Decoder(in_channels=512)


    def forward(self, x):
        x1, x2, x3, x4, bottleneck = self.encoder(x)
        bottleneck = F.interpolate(bottleneck, size=(256, 256), mode='bilinear', align_corners=False)
        
        print(f'Bottleneck shape before transformer: {bottleneck.shape}')

        bottleneck = self.transformer(bottleneck)

        print(f'Bottleneck shape after transformer: {bottleneck.shape}')

        logits = self.decoder(x1, x2, x3, x4, bottleneck)

        return logits

'''


if __name__ == "__main__":
    import torch

    x = torch.rand(1, 3, 512, 512)

    model = MyArchitecture_v2()

    logits = model(x)

    print(logits.shape)
