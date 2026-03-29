from tkinter import Variable
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum, flatten
from config import training_config
from loss.surface import simplex
import matplotlib.pyplot as plt

# class DiceLoss(nn.Module):
#     """
#       Dice Loss -> https://arxiv.org/pdf/1707.00478.pdf

#       input (Tensor): (N, C, H, W) - float / network outputs
#       target (Tensor): (N, H, W) - long / masks ground truth

#     """

#     def __init__(self) -> None:
#         super(DiceLoss, self).__init__()
#         self.eps: float = 1e-7

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         num_classes = input.shape[1]
#         true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(input, dim=1)
        
#         true_1_hot = true_1_hot.type(input.type())
#         dims = (0,) + tuple(range(2, target.ndimension()))
#         intersection = torch.sum(probas * true_1_hot, dims)
#         cardinality = torch.sum(probas + true_1_hot, dims)

#         dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        
#         return (1 - dice_loss)

class DiceLoss(nn.Module):
  """
    Dice Loss -> https://arxiv.org/pdf/1707.00478.pdf

    input (Tensor): (N, C, H, W) - float / network outputs
    target (Tensor): (N, H, W) - long / masks ground truth

  """

  def __init__(self) -> None:
    super(DiceLoss, self).__init__()
    self.eps: float = 1e-7
    self.classes = training_config['classes']

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    device = input.device
    
    # Convert target to one-hot encoding
    true_1_hot = torch.eye(self.classes, device=device)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

    probas = F.softmax(input, dim=1)
    
    # Compute intersection and union
    intersection = (probas * true_1_hot).sum(dim=(0, 2, 3))
    union = probas.sum(dim=(0, 2, 3)) + true_1_hot.sum(dim=(0, 2, 3))

    # Compute Dice loss
    dice_score = 2. * intersection / (union + self.eps)
    #dice_score = 2. * (intersection + self.eps) / (union + self.eps)
    dice_loss = 1. - dice_score.mean()
      
    return dice_loss

class GeneralizedDice():
  def __init__(self, **kwargs):
    self.idc: List[int] = [0, 1, 2]
    self.classes = training_config['classes']
    self.eps = 1e-7

  def __call__(self, input: Tensor, target: Tensor) -> Tensor:
    device = input.device
    true_1_hot = torch.eye(self.num_classes, device=device)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    
    probas = F.softmax(input, dim=1)
    
    # Flatten tensors
    input_flat = probas.view(probas.size(0), self.num_classes, -1)
    target_flat = true_1_hot.view(true_1_hot.size(0), self.num_classes, -1)
    
    # Calculate the Generalized Dice loss
    intersection = (input_flat * target_flat).sum(dim=-1)
    cardinality = (input_flat + target_flat).sum(dim=-1)

    class_weights = 1. / (target_flat.sum(dim=-1) ** 2 + self.eps)
    weighted_intersection = (intersection * class_weights).sum(dim=-1)
    weighted_union = (cardinality * class_weights).sum(dim=-1)

    dice_score = 2. * weighted_intersection / (weighted_union + self.eps)
    dice_loss = 1. - dice_score.mean()

    return dice_loss