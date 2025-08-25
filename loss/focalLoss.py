import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import training_config

# based on: https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/losses/focal.html

class FocalLoss(nn.Module):
  """
    FocalLoss -> https://arxiv.org/abs/1708.02002
    
    alpha (float): Weighting factor - [0, 1]
    gamma (float): Focusing parameter - gamma >= 0

    input (Tensor): (N, C, H, W) - float / network outputs
    target (Tensor): (N, H, W) - long / masks ground truth

  """

  def __init__(self, alpha: float = 1.0, gamma: float = 4.0) -> None:
    super(FocalLoss, self).__init__()
    self.alpha: float = alpha
    self.gamma: torch.Tensor = torch.tensor(gamma).cuda()
    self.eps: float = 1e-7
    self.classes = training_config['classes']

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    device = input.device
    
    # Apply softmax to the input
    input_soft = F.softmax(input, dim=1) + self.eps
    
    # Convert target to one-hot encoding
    target_one_hot = torch.eye(self.classes, device=device)[target.squeeze(1)].permute(0, 3, 1, 2).float().cuda()

    # compute the actual focal loss
    weight = (1 - input_soft) ** self.gamma
    focal = -self.alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    return (loss_tmp.mean())  