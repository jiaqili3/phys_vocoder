import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEndToEnd(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
    def forward(self, x):
        if x.size().__len__() != 3:
            x = x.reshape(1,1,-1)
        # pad
        segment_length = 32768*2
        if segment_length < x.size(-1):
            segment_length = segment_length * 2
        if segment_length < x.size(-1):
            segment_length = segment_length * 2
        assert segment_length >= x.size(-1)
        ori_size = x.size(-1)
        x = F.pad(x, (0, segment_length - x.size(-1)))
        return self.unet(x)[:,:,:ori_size]
    def load_model(self, checkpoint_path: str, device:str='cpu') -> None:
        self.unet.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"]["model"])
class UNetMelASVLoss(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetL1WavLoss(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMSEWavLoss(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetSpecAndWavLoss(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class TestUNet(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMelLossFinetuned(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMelLossEndToEnd(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetSpecLossEndToEnd(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMixedLossEndToEnd(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMixedLossNormalizedEndToEnd(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetMixedLoss1NormalizedEndToEnd(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetGAN(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
class UNetGANNormalized(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()

class UNetSpecLossEndToEnd1(UNetEndToEnd):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class UNet(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super().__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o
