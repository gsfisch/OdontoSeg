
from optimizers.radam import RAdam
from optimizers.ranger import Ranger
from optimizers.ralamb import Ralamb
from optimizers.rangerlars import RangerLars
from optimizers.novograd import Novograd

from torch import optim

"""
  Function to chooose an optmizer
"""


def optimizer(option='radam', model=None, lr=1e-5, weight_decay=0):

    if (option == 'adamax'):
        adamax = optim.Adamax
        return adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'sgd'):
        sgd = optim.SGD
        return sgd(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif (option == 'radam'):
        return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'adam'):
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'ranger'):
        return Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'ralamb'):
        return Ralamb(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'rangerlars'):
        return RangerLars(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif (option == 'novograd'):
        return Novograd(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NameError("Option is invalid. Got {}".format(option))
