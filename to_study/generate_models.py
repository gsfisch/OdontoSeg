import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from util.metrics import calculate_accuracy_eval, calculate_miou_eval
import numpy as np
import imageio.core.util
from skimage.io import imread, imsave
import pandas as pd
import warnings
from torch.serialization import SourceChangeWarning
import torch.nn.functional as F

def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())
    #return x.mul_(inner) if inplace else x.mul(inner)  # unexpected inplace issue with this


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)

def convert_relu_to_softplus(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
          setattr(model, child_name, Mish())
        else:
          convert_relu_to_softplus(child)

for index in tqdm.tqdm(range(30, 33)):
  experiment_name = 'experimento_' + str((index+1))
  with warnings.catch_warnings(record=True) as caught_warnings:
    model = torch.load('./experiments/' + experiment_name + '/model.pth')
  model.activation = nn.Identity()
  model.eval()
  print (model)
  # print (model)
  # convert_relu_to_softplus(model)
  # print (model)
  sample = torch.ones([1, 3, 512, 512]).to("cuda")
  traced_module = torch.jit.trace(model, sample)
  print ("dasd")
  torch.jit.save(traced_module, './experiments/' + experiment_name + '/model_new.pth')