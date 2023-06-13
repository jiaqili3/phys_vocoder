# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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

'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import pooling_layers
import torchaudio


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=80,
                 embed_dim=128,
                 num_class=1251,
                 pooling_func='TSTP',
                 two_emb_layer=True):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.torchfbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600,
            window_fn=torch.hamming_window, n_mels=feat_dim, normalized=True
        )
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, num_class)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, 0, :]
        # x = x * 32768
        # x = x / torch.max(torch.abs(x) + 1e-10)
        x = torchaudio.compliance.kaldi.fbank(
            waveform=x, sample_frequency=16000, frame_length=25, frame_shift=10, num_mel_bins=80, dither=0.0
        )
        x = x - torch.mean(x, dim=0)
        x = x / torch.sqrt(torch.var(x, dim=0) + 1e-8)

        x = torch.transpose(x, 0, 1)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            # return torch.tensor(0.0), embed_a
            return embed_a

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

    def encoding(self, x):
        return self.forward(x)


def ResNet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True, num_class=1251):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer,
                  num_class=num_class)


def ResNet50(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet101(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet152(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet221(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [6, 16, 48, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [10, 20, 64, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    from model_config import config as model_config

    model = ResNet34(**model_config['ResNet34'])
    model.eval()

    load_stat = torch.load('/home/wangli/ASGSR/pretrained_models/ResNet34/voxceleb_resnet34_LM.pt', map_location='cpu')
    # del [load_stat["projection.weight"]]
    # load_stat['torchfbank.spectrogram.window'] = model.state_dict()['torchfbank.spectrogram.window']
    # load_stat["torchfbank.mel_scale.fb"] = model.state_dict()["torchfbank.mel_scale.fb"]
    # print(load_stat)

    model.load_state_dict(load_stat)

    y = model(torch.randn(1, 1, 16000))
    print(y.size())

    # torch.save(model.state_dict(), '/home/wangli/ASGSR/pretrained_models/ResNet34/voxceleb_resnet34_LM.pt')
