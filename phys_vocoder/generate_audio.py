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
from hifigan.generator import HifiganEndToEnd
import os
from hifigan.dataset import MelDataset, LogMelSpectrogram

import sys
sys.path.append('..')
sys.path.append('.')

from phys_vocoder.unet.unet import UNetEndToEnd
import glob
import os

device = 'cuda:0'

def generate(args):
    generator = UNetEndToEnd().to(device)
    generator.eval()
    print("Loading checkpoint")
    generator.load_model(args.checkpoint, device=device)

    os.makedirs(args.out_dir, exist_ok=True)

    paths = glob.glob(str(args.in_dir))
    for path in paths:
        wav, _ = torchaudio.load(path)
        wav = wav.to(device)
        out = generator(wav).cpu()
        torchaudio.save(os.path.join(str(args.out_dir), str(Path(path).name)), out[0], 16000)
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
        # default='/home/lijiaqi/phys_vocoder/pretrained_models/model-145000.pt', # unet trained on xvec
        default='/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0702/model-45000.pt', # unet trained on ori
        # default='/mntcephfs/lab_data/lijiaqi/hifigan-checkpoints/0630/model-135000.pt', # hifigan trained on ori
        type=Path,
    )
    parser.add_argument(
        "--in_dir",
        metavar="in-dir",
        help="path to input directory containing the input wavs.",
        default='/mntcephfs/lab_data/lijiaqi/adver_out/ECAPATDNN_UNetEndToEnd_10_0.0004_0.005/PA_D_0000290_PA_E_0033977.wav',
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        metavar="out-dir",
        default="/home/lijiaqi/phys_vocoder/out",
        type=Path,
    )
    args = parser.parse_args()

    generate(args)
