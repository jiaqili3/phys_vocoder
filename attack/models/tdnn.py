# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
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

"""TDNN model for x-vector learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import pooling_layers
import torchaudio


class TdnnLayer(nn.Module):
    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """Define the TDNN layer, essentially 1-D convolution

        Args:
            in_dim (int): input dimension
            out_dim (int): output channels
            context_size (int): context size, essentially the filter size
            dilation (int, optional):  Defaults to 1.
            padding (int, optional):  Defaults to 0.
        """
        super(TdnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(self.in_dim,
                                 self.out_dim,
                                 self.context_size,
                                 dilation=self.dilation,
                                 padding=self.padding)

        # Set Affine=false to be compatible with the original kaldi version
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class XVEC(nn.Module):
    def __init__(self,
                 feat_dim=40,
                 hid_dim=512,
                 stats_dim=1500,
                 embed_dim=512,
                 num_class=1251,
                 pooling_func='TSTP',
                 **kwargs):
        """
        Implementation of Kaldi style xvec, as described in
        X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION
        """
        super(XVEC, self).__init__()
        self.feat_dim = feat_dim
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim

        self.torchfbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600,
            window_fn=torch.hamming_window, n_mels=feat_dim, normalized=True
        )
        self.frame_1 = TdnnLayer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_2 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_3 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_4 = TdnnLayer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = TdnnLayer(hid_dim,
                                 stats_dim,
                                 context_size=1,
                                 dilation=1)

        self.pool = getattr(pooling_layers, pooling_func)(in_dim=stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
        self.seg_2 = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        if len(x.size()) > 2 and x.size()[1] > 1:
            x = x[:, 0:1, :]  # first channel
        x = self.torchfbank(x) + 1e-6  # (batch,1, n_mels, time)
        x = x[:, 0, :, :]
        x = x.log()
        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)

        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        out = F.relu(embed_a)
        out = self.seg_bn_1(out)
        embed_b = self.seg_2(out)

        return embed_a, embed_b
        # return embed_b

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
        return self.forward(x)[0]


if __name__ == '__main__':
    model = XVEC(feat_dim=80, embed_dim=512, pooling_func='TSTP')
    model.eval()
    y = model(torch.rand(1, 1, 16000))
    print(y[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))
