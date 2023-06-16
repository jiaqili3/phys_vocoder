#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')
import attack.models.ResNetBlocks as ResNetBlocks


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class ResNetSE34V2(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE34V2, self).__init__()

        # print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input
        block = getattr(ResNetBlocks, block)

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window,
                                                 n_mels=n_mels)
        )

        outmap_size = int(self.n_mels / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def score(self, x):
        return self.forward(x)
    
    def forward(self, x):

        x = x[:, 0, :]
        with torch.cuda.amp.autocast(enabled=False):
            x = self.torchfb(x) + 1e-6
            if self.log_input: x = x.log()
            x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

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


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE34V2(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model


if __name__ == '__main__':
    load_stat = torch.load('/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.model', map_location='cpu')
    # select '__S__. in the load_stat, and then delete the '__S__.' in the key
    load_stat = {k.replace('__S__.', ''): v for k, v in load_stat.items()}

    model = MainModel(nOut=512, encoder_type='ASP', n_mels=64, log_input=True)
    model.eval()
    model_stat = model.state_dict()
    # select key,value in model_stat and load_stat, and then load the value to model_stat
    model_stat.update({k: v for k, v in load_stat.items() if k in model_stat})
    model.load_state_dict(model_stat)
    torch.save(model.state_dict(), '/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.pth')
