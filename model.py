import torch
import torch.nn as nn
from torchsummary import summary
from blocks import DoubleConvMaxPool, DoubleConv2d, UpDoubleConv, DoubleConvDropMaxPool, UpConvCropCopy
import math 

class tail(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(tail, self).__init__()
        self.dense = nn.Linear(580)
        self.bn = nn.BatchNorm1d(momentum=0.9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.dense2 = nn.Linear(3)

    def forward(self, x):
        x = dense(x)
        x = bn(x)
        x = relu(x)
        x = dropout(x)

        x = dense(x)
        x = bn(x)
        x = relu(x)
        x = dropout(x)

        x = dense2(x)
        
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
      
        self.down1 = DoubleConvMaxPool(3, 64, kernel_size=3, padding=1, stride=1)
        self.down2 = DoubleConvMaxPool(64, 128, kernel_size=3, padding=1, stride=1)
        self.down3 = DoubleConvMaxPool(128, 256, kernel_size=3, padding=1, stride=1)
        self.down4 = DoubleConvDropMaxPool(256, 512, kernel_size=3, padding=1, stride=1)
        self.down5 = DoubleConv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.drop = nn.Dropout(0.5)
        self.up1 = UpDoubleConv(1024, 512, kernel_size=3, stride=1)
        self.up2 = UpDoubleConv(512, 256, kernel_size=3, stride=1)
        self.up3 = UpDoubleConv(256, 128, kernel_size=3, stride=1)
        self.up4 = UpDoubleConv(128, 64, kernel_size=3, stride=1)

        self.out = nn.Conv2d(64, 3, 1)

        self._initialize_weights()
    
    def forward(self, x):
        conv1, pool  = self.down1(x)
        conv2, pool  = self.down2(pool)
        conv3, pool  = self.down3(pool)
        conv4, pool  = self.down4(pool)
        
        x = self.down5(pool)
        x = self.drop(x)

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)

        x = self.out(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# gg = Unet()
# gg.cuda()
# summary(gg, input_size=(3, 512, 512))