from .inception import inception_v3
from .BN_Inception import BNInception
from .resnet_50 import resnet_50

__factory = {
    'bn': BNInception,
    'inception': inception_v3,
    'resnet_50': resnet_50,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)