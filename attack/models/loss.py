import torch
import torch.nn as nn

def verification_loss(score, threshold, y):
    loss = score - threshold
    return loss

class SpeakerVerificationLoss(nn.Module):
    def __init__(self, targeted=False, threshold=None):
        super().__init__()
        self.targeted = targeted
        self.threshold = threshold

    def forward(self, scores, label):
        device = scores.device
        label = label.to(device)
        loss = verification_loss(scores, self.threshold, label)
        return loss