import torch
from torch import Tensor, einsum
from typing import Iterable,Set
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
import timeit

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)

    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # assert one_hot(torch.Tensor(seg))
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1, 2])

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

class GeneralizedDice(nn.Module):
    def __init__(self, idc):
        super(GeneralizedDice, self).__init__()
        self.idc: List[int] = idc
        self.eps: float = 1e-7


    def forward(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:       
        probs = F.softmax(probs, dim=1).cuda() + self.eps
        target = torch.eye(3)[target.squeeze(1)]
        target = target.permute(0, 3, 1, 2).float().cuda()

        # assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        values = einsum("bcwh->bc", tc)
        indexes = []

        for i in range(2):
          for j in range(3):
            if (values[i][j] == 0):
              indexes.append(j)

        w[0][indexes[0]] = 0
        w[1][indexes[1]] = 0

        # print ("\n")
        # print (einsum("bcwh->bc", tc))
        # print (w)
            
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2*(einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()
      
        return loss

class SurfaceLoss(nn.Module):
    def __init__(self, idc):
        super(SurfaceLoss, self).__init__()
        self.idc: List[int] = idc

    def forward(self, probs: Tensor, masks: Tensor, _: Tensor) -> Tensor:
        new_masks = class2one_hot(masks.clone(), 3)
        
        for i in range(2):
          dist_maps = class2one_hot(masks[0], 3)
          dist_maps = torch.tensor(one_hot2dist(dist_maps[0].cpu().numpy()), dtype=torch.float32).cuda()
          new_masks[i] = dist_maps

        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = new_masks[:, self.idc, ...].type(torch.float32)
        multipled = einsum("bcwh,bcwh->bc", pc, dc).cuda()
        
        loss = (multipled.sum()) / (2.0 * 512 * 512 )

        return loss