# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from asteroid_filterbanks import Encoder, ParamSincFB
import sys
import torch.nn.functional as F

from .RawNetBasicBlock import PreEmphasis, Bottle2neck


class RawNet3(nn.Module):
    def __init__(self, block, model_scale, context, summed, C=1024, **kwargs):
        super().__init__()

        nOut = kwargs["nOut"]

        self.context = context
        self.encoder_type = kwargs["encoder_type"]
        self.log_sinc = kwargs["log_sinc"]
        self.norm_sinc = kwargs["norm_sinc"]
        self.out_bn = kwargs["out_bn"]
        self.summed = summed
        block = Bottle2neck

        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv1 = Encoder(
            ParamSincFB(
                C // 4,
                251,
                stride=kwargs["sinc_stride"],
            )
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = block(
            C // 4, C, kernel_size=3, dilation=2, scale=model_scale, pool=5
        )
        self.layer2 = block(
            C, C, kernel_size=3, dilation=3, scale=model_scale, pool=3
        )
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536
        if self.encoder_type == "ECA":
            attn_output = 1536
        elif self.encoder_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)

        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

        self.mp3 = nn.MaxPool1d(3)
    
    def score(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """
        x = x[:, 0, :]
        with torch.cuda.amp.autocast(enabled=False):
            x = self.preprocess(x)
            x = torch.abs(self.conv1(x))
            if self.log_sinc:
                x = torch.log(x + 1e-6)
            if self.norm_sinc == "mean":
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std(x, dim=-1, keepdim=True)
                s[s < 0.001] = 0.001
                x = (x - m) / s

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(self.mp3(x1) + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        if self.context:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, 1, t),
                ),
                dim=1,
            )
        else:
            global_x = x

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4, max=1e4)
        )

        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)

        x = self.fc6(x)

        if self.out_bn:
            x = self.bn6(x)

        return x

    def encoding(self, x):
        return self.forward(x)

    def make_decision_SV(self, x1, x2):
        emb1 = self.score(x1)
        emb2 = self.score(x2)

        cos = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-6)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        decisions = torch.where(cos > self.threshold, ones, zeros)
        return decisions, cos


if __name__ == '__main__':
    from model_config import config as model_config

    model = RawNet3(**model_config['RawNet3'])

    load_stat = torch.load('/home/wangli/ASGSR/pretrained_models/RawNet3/model.pt', map_location='cpu')
    load_stat = load_stat['model']

    model.eval()
    model.load_state_dict(load_stat)

    torch.save(model.state_dict(), '/home/wangli/ASGSR/pretrained_models/RawNet3/model.pt')
    x = torch.randn(1, 1, 16000)
    y = model(x)
    print(y.shape)
