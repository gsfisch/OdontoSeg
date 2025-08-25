import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


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

def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())
    #return x.mul_(inner) if inplace else x.mul(inner)  # unexpected inplace issue with this


def convert_relu_to_swish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
          setattr(model, child_name, MemoryEfficientSwish())
        elif (child_name == '_swish'):
          setattr(model, child_name, MemoryEfficientSwish())
        else:
          convert_relu_to_swish(child)


def convert_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish().cuda())
        elif (child_name == '_swish'):
            # a = 2
            setattr(model, child_name, nn.Identity())
        else:
            convert_relu_to_mish(child)


def convert_swith_to_identity(model):
    for child_name, child in model.named_children():
        if (child_name == '_swish'):
            setattr(model, child_name, nn.ReLU().cuda())
        else:
            convert_swith_to_identity(child)


# class_predicted = {
#   0:(0,0,255),
#   1:(0,255,0),
#   2:(255,0,0)
# }

# mapping = [
#   [  0,   0,   255],
#   [  0,   255,   0],
#   [  255,   0,  0],
# ]
