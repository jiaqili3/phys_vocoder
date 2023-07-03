from pathlib import Path
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm
from hifigan.dataset import LogMelSpectrogram
import argparse
import logging
from pathlib import Path
import torch.nn.functional as F
import torch
from hifigan.generator import HifiganGenerator, HifiganEndToEnd

from hifigan.dataset import MelDataset, LogMelSpectrogram

import sys
sys.path.append('..')
sys.path.append('.')

from phys_vocoder.unet.unet import UNet
import glob
import os

device = 'cuda:0'

def generate(args):
    generator = UNet().to(device)
    generator.eval()
    print("Loading checkpoint")
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device)["generator"]["model"])
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # generator.load_state_dict(checkpoint["generator"]["model"])

    paths = glob.glob(str(args.in_dir))
    for path in paths:
        wav, _ = torchaudio.load(path)
        wav = wav.reshape(1,1,-1)
        # pad
        segment_length = 32768*2
        if segment_length < wav.size(-1):
            segment_length = segment_length * 2
        if segment_length < wav.size(-1):
            segment_length = segment_length * 2
        assert segment_length >= wav.size(-1)
        ori_size = wav.size(-1)
        wav = F.pad(wav, (0, segment_length - wav.size(-1)))
        wav = wav.to(device)
        wavs = generator(wav).cpu()
        torchaudio.save(os.path.join(str(args.out_dir), str(Path(path).name)), wavs[0,:,:ori_size], 16000)
        print(f'saved to {os.path.join(str(args.out_dir), str(Path(path).name))}')


    # model_name = f"hifigan_hubert_{args.model}" if args.model != "base" else "hifigan"
    # hifigan = torch.hub.load("bshall/hifigan:main", model_name).cuda()

    # print(f"Generating audio from {args.in_dir}")
    # for path in tqdm(list(args.in_dir.rglob("*.npy"))):
    #     mel = torch.from_numpy(np.load(path))
    #     mel = mel.unsqueeze(0).cuda()

    #     wav, sr = hifigan.generate(mel)
    #     wav = wav.squeeze(0).cpu()

    #     out_path = args.out_dir / path.relative_to(args.in_dir)
    #     out_path.parent.mkdir(exist_ok=True, parents=True)
    #     torchaudio.save(out_path.with_suffix(".wav"), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio for a directory of 16kHz audios using HiFi-GAN."
    )
    parser.add_argument(
        "--checkpoint",
        default='/mnt/workspace/lijiaqi/unet_checkpoints/0702/model-55000.pt',
        type=Path,
    )
    parser.add_argument(
        "--in_dir",
        metavar="in-dir",
        help="path to input directory containing the input wavs.",
        default='/mnt/workspace/lijiaqi/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10/dev/*.wav',
        # default='/mnt/workspace/lijiaqi/hifigan/PA_D_0000001.wav_synthesis.wav',
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        metavar="out-dir",
        default="/mnt/workspace/lijiaqi/hifigan/out",
        type=Path,
    )
    args = parser.parse_args()

    generate(args)
