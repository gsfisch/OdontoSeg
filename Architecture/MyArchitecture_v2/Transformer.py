import timm
import torch.nn as nn



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = timm.create_model('tiny_vit_21m_512', pretrained=True, features_only=True)


    def forward(self, x):
        x = self.transformer(x)

        return x


if __name__ == "__main__":
    import torch

    img = torch.rand(1, 3, 512, 512)

    model = Transformer()

    feature_list = model(img)
    # 96, 128, 128
    # 192, 64, 64
    # 384, 32, 32
    # 576, 16, 16

    for feature_map in feature_list:
        print(feature_map.shape)
