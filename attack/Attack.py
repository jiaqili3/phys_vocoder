import torch
import sys
from attack.attacks.PGD import PGD

class PGDAttack(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.pgd = PGD(**kwargs)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y):
        # returns: adver, success
        return self.pgd.attack_sv(x1, x2, y)

