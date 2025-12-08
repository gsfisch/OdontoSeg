import timm
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = timm.create_model('pvt_v2_b1', pretrained=True, features_only=True)


    def forward(self, x):
        x = self.transformer(x)

        return x


if __name__ == "__main__":
    import torch

    img = torch.rand(1, 3, 512, 512)

    model = Transformer()

    feature_list = model(img)

    for feature_map in feature_list:
        print(feature_map.shape)
