import torch

def get_class_weigths():
    weights = [0.14647541984259294, 1.4230167810174439, 1]
    class_weights = torch.FloatTensor(weights).cuda()

    return class_weights