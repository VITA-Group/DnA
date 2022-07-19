import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from PIL import Image
import random
import numpy as np


class subsetCIFAR10(CIFAR10):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()


class subsetCIFAR100(CIFAR100):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()
