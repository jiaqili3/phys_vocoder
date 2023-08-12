# adopted from https://github.com/jik876/hifi-gan/blob/master/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
# from .mpd import MultiPeriodDiscriminator
from .msstftd import MultiScaleSTFTDiscriminator
LRELU_SLOPE = 0.1

def get_padding(k, d):
    return int((k * d - d) / 2)

class PeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Period Discriminator"""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            [Tensor]: discriminator scores per sample in the batch.
            [List[Tensor]]: list of features from each convolutional layer.
        """
        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(2),
                PeriodDiscriminator(3),
                PeriodDiscriminator(5),
                PeriodDiscriminator(7),
                PeriodDiscriminator(11),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            [List[Tensor]]: list of scores from each discriminator.
            [List[List[Tensor]]]: list of features from each discriminator's convolutional layers.
        """
        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats

class UNetDiscriminator(nn.Module):
    """HiFiGAN discriminator"""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msstftd = MultiScaleSTFTDiscriminator()

    def forward(
        self, y: torch.Tensor, y_hat
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            y (Tensor): input waveform.
        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of features from from each discriminator's convolutional layers.
        """
        scores, feats = self.mpd(y)
        scores_, feats_ = self.mpd(y_hat)
        scores1, scores2, feats1, feats2 = self.msstftd(y, y_hat)
        return scores + scores1, scores_ + scores2, feats + feats1, feats_ + feats2


def feature_loss(
    features_real: List[List[torch.Tensor]], features_generate: List[List[torch.Tensor]]
) -> float:
    loss = 0
    for r, g in zip(features_real, features_generate):
        for rl, gl in zip(r, g):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(real, generated):
    loss = 0
    real_losses = []
    generated_losses = []
    for r, g in zip(real, generated):
        r_loss = torch.mean((1 - r) ** 2)
        g_loss = torch.mean(g ** 2)
        loss += r_loss + g_loss
        real_losses.append(r_loss.item())
        generated_losses.append(g_loss.item())

    return loss, real_losses, generated_losses


def generator_loss(discriminator_outputs):
    loss = 0
    generator_losses = []
    for x in discriminator_outputs:
        l = torch.mean((1 - x) ** 2)
        generator_losses.append(l)
        loss += l

    return loss, generator_losses
