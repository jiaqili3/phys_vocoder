from initializer import config
import torch
from torch.utils.data import DataLoader
import os

from attack.attacks.pgd import PGD

device = 'cuda:0'

model = config.model
model = model.to(device)
phys_vocoder = config.phys_vocoder_model
phys_vocoder = phys_vocoder.to(device)

class CombinedModel(torch.nn.Module):
    def __init__(self, model, phys_vocoder):
        super(CombinedModel, self).__init__()
        self.model = model
        self.phys_vocoder = phys_vocoder
    def forward(self, x1, x2):
        # x1 enroll, x2 test
        # return (decisions, cos sim)
        x2 = self.phys_vocoder(x2)
        return self.model.make_decision_sv(x1,x2)

combined_model = CombinedModel(model, phys_vocoder)

attacker = PGD(combined_model)
attacker = attacker.to(device)