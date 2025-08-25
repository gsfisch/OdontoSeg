import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum

class JaccardLoss(nn.Module):
    """
      Jaccard Loss -> https://arxiv.org/pdf/1705.08790.pdf 

      input (Tensor): (N, C, H, W) - float / network outputs
      target (Tensor): (N, H, W) - long / masks ground truth

    """

    def __init__(self) -> None:
        super(JaccardLoss, self).__init__()
        self.eps: float = 1e-7

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      num_classes = input.shape[1]

      true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
      true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
      probas = F.softmax(input, dim=1)
      true_1_hot = true_1_hot.type(input.type())

      dims = (0,) + tuple(range(2, target.ndimension()))
      intersection = torch.sum(probas * true_1_hot, dims) 
      cardinality = torch.sum(probas + true_1_hot, dims)
      union = cardinality - intersection

      jacc_loss = ((intersection / (union + self.eps))).mean()

      return (1 - jacc_loss)