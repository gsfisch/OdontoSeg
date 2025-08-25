from util.utils import extract_img, extract_img_new
import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride)
        # self.batchNorm = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchNorm(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        # x = self.batchNorm(x)
        x = self.relu(x)

        return x 

class DoubleConvDropMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DoubleConvDropMaxPool, self).__init__()
        self.doubleConv = DoubleConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.dropout = nn.Dropout(0.5)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        conv = self.doubleConv(x)
        drop = self.dropout(conv)
        pool = self.maxPool(drop)
        
        return drop, pool 

class DoubleConvMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DoubleConvMaxPool, self).__init__()
        self.doubleConv = DoubleConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        conv = self.doubleConv(x)
        pool = self.maxPool(conv)
        
        return conv, pool 


class UpConvTransposeCropCopy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpConvTransposeCropCopy, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)       

    def forward(self, x1, oldConv):
        x1 = self.upConv(x1)

        # x1 = extract_img_new(x1, oldConv)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, oldConv)
        x1 = torch.cat((x1, x2), dim=1)

        return x1 

@torch.jit.script
def center_crop_helper(layer, xy1: int, xy2: int, max_height: int, max_width: int):
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class UpConvCropCopy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpConvCropCopy, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)    
        self.relu = nn.ReLU()

    def center_crop(self, layer, max_height, max_width):
      _, _, h, w = layer.size()
      xy1 = (w - max_width) // 2
      xy2 = (h - max_height) // 2
      return center_crop_helper(layer, xy1, xy2, max_height, max_width)

    def _crop_concat(self, bypass, upsampled):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """

       

        c = (bypass.size()[2] - upsampled.size()[2]) // 2

        bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((bypass, upsampled), dim=1)



    def forward(self, x, oldConv):
        x = self.upConv(x)
        x = self.relu(x)
        x = self.center_crop(x, x.size(2), x.size(3))

        x = torch.cat((x, oldConv), dim=1)
        
        return x

class UpDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpDoubleConv, self).__init__()
        self.UpConvCropCopy = UpConvCropCopy(in_channels, out_channels, kernel_size=2, stride=2)
        self.doubleConv = DoubleConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
    def forward(self, conv, oldConv):
        up = self.UpConvCropCopy(conv, oldConv)
        k = self.doubleConv(up)
        
        return k 