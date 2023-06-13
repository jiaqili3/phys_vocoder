# Copyright (c) 2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' This implementation is adapted from github repo:
    https://github.com/lawlict/ECAPA-TDNN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
from model.BasicModel import BasicModel
from model import pooling_layers

''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(self,
                 channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(self.width,
                          self.width,
                          kernel_size,
                          stride,
                          padding,
                          dilation,
                          bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation,
                 scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0),
            Res2Conv1dReluBn(channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             scale=scale),
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0),
            SE_Connect(channels))

    def forward(self, x):
        return x + self.se_res2block(x)


class ECAPA_TDNN(BasicModel):
    def __init__(self, channels=512, feat_dim=80, embed_dim=192, num_class=1251, pooling_func='ASTP', global_context_att=False, augment=False,
                 **kwargs):
        super().__init__(augment=augment)

        self.torchfbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600,
            window_fn=torch.hamming_window, n_mels=feat_dim, normalized=True
        )
        self.layer1 = Conv1dReluBn(feat_dim,
                                   channels,
                                   kernel_size=5,
                                   padding=2)
        self.layer2 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=2,
                                   dilation=2,
                                   scale=8)
        self.layer3 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=3,
                                   dilation=3,
                                   scale=8)
        self.layer4 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=4,
                                   dilation=4,
                                   scale=8)

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.linear2 = nn.Linear(embed_dim, num_class)
        self.augment = augment

    def forward(self, x):
        if len(x.size()) > 2 and x.size()[1] > 1:
            x = x[:, 0:1, :]  # first channel
        x = self.torchfbank(x) + 1e-6  # (Batch, channel, n_mels, time)
        if self.augment:
            x = self.spec_augment(x)
        x = x[:, 0, :, :]
        x = x.log()
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pool(out))
        out = self.linear(out)
        embed_a = out
        out = F.relu(out)
        out = self.bn2(out)
        out = self.linear2(out)
        embed_b = out

        return embed_a, embed_b

    def score(self, x):
        return self.forward(x)

    def make_decision(self, x):
        _, scores = self.score(x)
        decisions = torch.argmax(scores, dim=1)
        return decisions, scores

    def make_decision_SV(self, x1, x2):
        emb1, _ = self.score(x1)
        emb2, _ = self.score(x2)

        cos = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-6)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        decisions = torch.where(cos > self.threshold, ones, zeros)
        return decisions, cos


def ECAPATDNNc1024(feat_dim, embed_dim, pooling_func='ASTP'):
    return ECAPA_TDNN(channels=1024,
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      pooling_func=pooling_func)


def ECAPATDNNGLOBc1024(feat_dim, embed_dim, num_class, pooling_func='ASTP', augment=False):
    return ECAPA_TDNN(channels=1024,
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      num_class=num_class,
                      pooling_func=pooling_func,
                      global_context_att=True, augment=augment)


def ECAPATDNNc512(feat_dim, embed_dim, pooling_func='ASTP'):
    return ECAPA_TDNN(channels=512,
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      pooling_func=pooling_func)


def ECAPATDNNGLOBc512(feat_dim, embed_dim, num_class, pooling_func='ASTP'):
    return ECAPA_TDNN(channels=512,
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      num_class=num_class,
                      pooling_func=pooling_func,
                      global_context_att=True)


if __name__ == '__main__':
    x = torch.randn(1, 1, 64000)
    # x,_ = torchaudio.load('/home/wangli/ASGSR/test.wav')
    # x = x.unsqueeze(0)
    model = ECAPATDNNGLOBc1024(feat_dim=80, embed_dim=192, num_class=1251, pooling_func='TSTP', augment=True)
    model.eval()
    cls = model.score(x)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
