from initializer import config
import torch
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    attack_class = config.attack.attack_class
    attack_class = attack_class(**config.attack)