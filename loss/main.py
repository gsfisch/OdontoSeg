import torch
import torch.nn.functional as F
import torch.nn as nn
from loss.focalLoss import FocalLoss
from loss.dice import DiceLoss, GeneralizedDice
from loss.kappa import quadratic_weighted_kappa
from loss.jaccard import JaccardLoss
from loss.surface import SurfaceLoss, GeneralizedDice as GD
from torch.autograd import Variable
from config import training_config

"""
  Function to chooose a loss function
"""


def criterion(option='ce', outputs=None, masks=None):
    class_weights = torch.FloatTensor(training_config['class_weigths']).cuda()
        
    if (option == 'focal'):
        return FocalLoss(alpha=1)(outputs, masks)
    elif (option == 'ce'):
        return F.cross_entropy(outputs, masks, weight=class_weights)
    elif (option == 'grdice'):
        return GeneralizedDice()(outputs, masks)
    elif (option == 'dice'):
        return DiceLoss()(outputs, masks)
    # elif (option == 'jaccard'):
    #     return JaccardLoss()(outputs, masks)
    # elif (option == 'kappa'):
    #     return quadratic_weighted_kappa(outputs, masks)
    # elif (option == 'surface'):
    #     outputs = F.softmax(outputs, dim=1)
    #     # surface_loss = SurfaceLoss(idc=[0, 1, 2])
    #     return GD(idc=[0, 1, 2])(outputs, masks, None)
    #     # + SurfaceLoss(idc=[1, 2])(outputs, masks, None)
    # elif (option == 'combo'):
    #     return FocalLoss(alpha=1)(outputs.clone(), masks.clone()) + GD(idc=[0, 1, 2])(outputs.clone(), masks.clone(), None) 
    # elif (option == 'combos'):
    #     cross = F.cross_entropy(F.softmax(outputs.clone(), dim=1), masks.clone(), reduction='none')
    #     jaccard = JaccardLoss()(outputs.clone(), masks.clone())
    #     dice = DiceLoss()(outputs.clone(), masks.clone())
    #     targetOnedD = masks.clone().view(2, -1)
    #     weights = []

    #     for batch in range(2):
    #         for classe in range(3):
    #             num_pixel = ((targetOnedD[batch] == classe)).float().sum()

    #         if (num_pixel == 0):
    #             weights.append(classe)
    #             break

    #     newvec = masks.clone()

    #     newvec[0][masks[0] == weights[0]] = 0
    #     newvec[0][masks[0] != weights[0]] = 1
    #     newvec[1][masks[1] == weights[1]] = 0
    #     newvec[1][masks[1] != weights[1]] = 1

    #     return (newvec*cross).mean() + DiceLoss()(outputs.clone(), masks.clone()), \
    #             (newvec*cross).mean(), jaccard, dice
    else:
        raise NameError("Option is invalid. Got {}".format(option))
