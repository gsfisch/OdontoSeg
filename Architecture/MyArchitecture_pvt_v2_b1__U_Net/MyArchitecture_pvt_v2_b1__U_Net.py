import torch.nn as nn
import torch.nn.functional as F
from Architecture.MyArchitecture_pvt_v2_b1__U_Net.Encoder import Encoder
from Architecture.MyArchitecture_pvt_v2_b1__U_Net.Decoder import Decoder
from Architecture.MyArchitecture_pvt_v2_b1__U_Net.Transformer import Transformer
#from Encoder import Encoder
#from Decoder import Decoder
#from Transformer import Transformer


class MyArchitecture_pvt_v2_b1__U_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(in_channels=3, encoder_channels=[64, 128, 320, 512, 1024])
        self.transformer = Transformer()
        self.decoder = Decoder(bottleneck_channels=1024, skip_channels=[64, 128, 320, 512])


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




if __name__ == "__main__":
    import torch

    x = torch.rand(1, 3, 512, 512)

    model = MyArchitecture_pvt_v2_b1__U_Net()

    logits = model(x)

    print(logits.shape)
