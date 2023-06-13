from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as transforms


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel
    def forward_nopad(self, wav):
        # wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


class MelDataset(Dataset):
    def __init__(
        self,
        ori_wavs_dir: Path,
        exp_wavs_dir: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        finetune: bool = False,
    ):
        self.ori_wavs_dir = ori_wavs_dir
        # self.mels_dir = ori_wavs_dir / "mels"
        self.exp_wavs_dir = exp_wavs_dir
        # self.data_dir = self.wavs_dir if not finetune else self.mels_dir

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.finetune = finetune

        suffix = ".wav"
        # suffix = ".wav" if not finetune else ".npy"
        pattern = f"dev/*{suffix}" if not train else f"*{suffix}"

        self.ori_metadata = [
            path.relative_to(self.ori_wavs_dir).with_suffix("")
            for path in self.ori_wavs_dir.rglob(pattern)
        ]
        self.exp_metadata = [
            path.relative_to(self.exp_wavs_dir).with_suffix("")
            for path in self.exp_wavs_dir.rglob(pattern)
        ]

        self.logmel = LogMelSpectrogram()

    def __len__(self):
        return len(self.ori_metadata)

    def __getitem__(self, index):
        path = self.ori_metadata[index]
        ori_wav_path = self.ori_wavs_dir / path
        path = self.exp_metadata[index]
        exp_wav_path = self.exp_wavs_dir / path

        info = torchaudio.info(ori_wav_path.with_suffix(".wav"))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )
        info = torchaudio.info(exp_wav_path.with_suffix(".wav"))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        if self.finetune:
            # mel_path = self.mels_dir / path
            # src_logmel = torch.from_numpy(np.load(mel_path.with_suffix(".npy")))
            # src_logmel = src_logmel.unsqueeze(0)
            wav, _ = torchaudio.load(
                filepath=ori_wav_path.with_suffix(".wav"),
            )
            src_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

            mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
            mel_offset = random.randint(0, max(mel_diff, 0))

            frame_offset = self.hop_length * mel_offset
        else:
            frame_diff = info.num_frames - self.segment_length
            frame_offset = random.randint(0, max(frame_diff, 0))

        wav, _ = torchaudio.load(
            filepath=exp_wav_path.with_suffix(".wav"),
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        # if not self.finetune and self.train:
        # if self.train:
        #     gain = random.random() * (0.99 - 0.4) + 0.4
        #     flip = -1 if random.random() > 0.5 else 1
        #     wav = flip * gain * wav / wav.abs().max()

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if self.finetune:
            if self.train:
                src_logmel = src_logmel[
                    :, :, mel_offset : mel_offset + mel_frames_per_segment
                ]

            if src_logmel.size(-1) < mel_frames_per_segment:
                src_logmel = F.pad(
                    src_logmel,
                    (0, mel_frames_per_segment - src_logmel.size(-1)),
                    "constant",
                    src_logmel.min(),
                )
        else:
            src_logmel = tgt_logmel.clone()

        return wav, src_logmel, tgt_logmel
