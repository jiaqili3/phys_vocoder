from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import glob
import pdb

from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as transforms

class WavDataset(Dataset):
    def __init__(
        self,
        ori_wavs_dir: str,
        exp_wavs_dir: str,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        eval_size=500,
    ):
        # self.ori_wavs_dir = glob.glob(ori_wavs_dir)
        # self.mels_dir = ori_wavs_dir / "mels"
        # self.exp_wavs_dir = exp_wavs_dir

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.finetune = True

        # suffix = ".wav"
        # suffix = ".wav" if not finetune else ".npy"
        # pattern = f"dev/*{suffix}" if not train else f"*{suffix}"

        random.seed(0)
        self.ori_metadata = glob.glob(ori_wavs_dir)
        random.shuffle(self.ori_metadata)
        self.ori_metadata = self.ori_metadata[:eval_size] if not train else self.ori_metadata[eval_size:]
        random.seed(0)
        self.exp_metadata = glob.glob(exp_wavs_dir)
        random.shuffle(self.exp_metadata)
        self.exp_metadata = self.exp_metadata[:eval_size] if not train else self.exp_metadata[eval_size:]
        assert len(self.ori_metadata) == len(self.exp_metadata)
        for i in range(len(self.ori_metadata)):
            assert self.ori_metadata[i].split('/')[-1] == self.exp_metadata[i].split('/')[-1]

    def __len__(self):
        return len(self.ori_metadata)

    def __getitem__(self, index):
        # path = self.ori_metadata[index]
        ori_wav_path = self.ori_metadata[index]
        # path = self.exp_metadata[index]
        exp_wav_path = self.exp_metadata[index]

        info = torchaudio.info(ori_wav_path)
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )
        info = torchaudio.info(exp_wav_path)
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        # mel_path = self.mels_dir / path
        # src_logmel = torch.from_numpy(np.load(mel_path.with_suffix(".npy")))
        # src_logmel = src_logmel.unsqueeze(0)
        src_wav, _ = torchaudio.load(
            filepath=ori_wav_path,
        )
        tgt_wav, _ = torchaudio.load(
            filepath=exp_wav_path,
        )
        assert src_wav.size() == tgt_wav.size()
        frame_diff = src_wav.shape[-1] - self.segment_length
        frame_offset = random.randint(0, max(frame_diff, 0))
        # The input of the model should be fixed length.
        # if src_wav.size(-1) % self.segment_length != 0:
        #     padded_length = self.segment_length - (src_wav.size(-1) % self.segment_length)
        #     src_wav = torch.cat([src_wav, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
        #     tgt_wav = torch.cat([tgt_wav, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

        if src_wav.size(-1) <= self.segment_length:
            src_wav = F.pad(src_wav, (0, self.segment_length - src_wav.size(-1)))
            tgt_wav = F.pad(tgt_wav, (0, self.segment_length - tgt_wav.size(-1)))
        else:
            src_wav = src_wav[:, frame_offset:frame_offset+self.segment_length]
            tgt_wav = tgt_wav[:, frame_offset:frame_offset+self.segment_length]

        # if not self.finetune and self.train:
        # if self.train:
        #     gain = random.random() * (0.99 - 0.4) + 0.4
        #     flip = -1 if random.random() > 0.5 else 1
        #     wav = flip * gain * wav / wav.abs().max()

        return src_wav, src_wav, tgt_wav
