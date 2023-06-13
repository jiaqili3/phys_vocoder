import torch.nn as nn
import torchvision
import abc


class BasicModel(nn.Module):
    def __init__(self, augment=False):
        super().__init__()
        if augment:
            self.spec_augment = torchvision.transforms.RandomErasing(p=0.5, scale=(0.05, 0.1), ratio=(0.5, 1.5), value=1e-6, inplace=False)

    @abc.abstractmethod
    def score(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def make_decision(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def make_decision_SV(self, x1, x2):
        raise NotImplementedError
