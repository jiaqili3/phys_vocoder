import torchaudio
import torchaudio.transforms as transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils.utils import compute_PESQ
import argparse
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import random

from phys_vocoder.dataset import WavDataset
import pdb
# from hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram

from phys_vocoder.unet.unet import UNet
from pymcd.mcd import Calculate_MCD

FOLDER = 'L2WavLoss'
random.seed(16)
def extract_mcd(audio1, audio2, mode='plain'):
    """Extract Mel-Cepstral Distance for a two given audio.
    Args:
        audio1: The given reference audio. It is an audio path.
        audio2: The given synthesized audio. It is an audio path.
        mode: "plain", "dtw" and "dtw_sl".
    """
    mcd_toolbox = Calculate_MCD(MCD_mode=mode)
    mcd_value = mcd_toolbox.calculate_mcd(audio1, audio2)

    return mcd_value

BATCH_SIZE = 32
SEGMENT_LENGTH = 32768*2
HOP_LENGTH = 160
SAMPLE_RATE = 16000
BASE_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 2e-4
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.999
WEIGHT_DECAY = 1e-5
EPOCHS = 1300
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 2500
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 5000

rank = "cuda:0"
def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.tight_layout()
    return fig

def eval_spec(args):
    validation_dataset = WavDataset(
            ori_wavs_dir=args.dataset_dir,
            exp_wavs_dir=args.exp_wavs_dir,
            segment_length=SEGMENT_LENGTH,
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            train=False,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
    )
    generator = UNet().to(rank)
    checkpoint = torch.load(args.resume, map_location="cuda:0")
    generator.load_state_dict(checkpoint["generator"]["model"])
    generator.eval()
    average_mcd = 0
    for j, (_, src, tgt) in enumerate(validation_loader, 1):
        src, tgt = src.to(rank), tgt.to(rank)

        with torch.no_grad():
            out = generator(src)
        # pdb.set_trace()
        os.makedirs(f'tmp/{FOLDER}', exist_ok=True)
        torchaudio.save(f'tmp/{FOLDER}/1.wav', src.squeeze(0).cpu(), 16000, encoding='PCM_S', bits_per_sample=16)
        torchaudio.save(f'tmp/{FOLDER}/2.wav', out.squeeze(0).cpu(), 16000, encoding='PCM_S', bits_per_sample=16)
        mcd = extract_mcd(f'tmp/{FOLDER}/1.wav', f'tmp/{FOLDER}/2.wav')
        # print(mcd)
        # validation_loss = F.l1_loss(out, tgt)
        # change loss
        # mcd = compute_PESQ(out.reshape(-1).cpu().numpy(), tgt.reshape(-1).cpu().numpy())
        # spec_out = spectrogram(out)
        # spec_tgt = spectrogram(tgt)
        # spec_out = spec_out.squeeze(0).squeeze(0)
        # spec_tgt = spec_tgt.squeeze(0).squeeze(0)
        average_mcd += (
            mcd - average_mcd
        ) / j

        # spec_diff = spec_out - spec_tgt
        # writer.add_figure(
        #     f"spec_diff/spec_{j}",
        #     plot_spectrogram(spec_diff.cpu().numpy()),
        # )
        # writer.add_figure(
        #     f"spec_out/spec_{j}",
        #     plot_spectrogram(spec_out.cpu().numpy()),
        # )
        # writer.add_figure(
        #     f"spec_tgt/spec_{j}",
        #     plot_spectrogram(spec_tgt.cpu().numpy()),
        # )
        # writer.add_audio(
        #     f"out/wav_{j}",
        #     out,
        #     sample_rate=16000,
        # )
        # writer.add_audio(
        #     f"tgt/wav_{j}",
        #     out,
        #     sample_rate=16000,
        # )

        

        # if average_spec_diff is None:
        #     average_spec_diff = torch.sum(spec_diff, dim=1)
        # else:
        #     average_spec_diff += (
        #         torch.sum(spec_diff, dim=1) - average_spec_diff
        #     ) / j
        print(j, average_mcd)
    print(FOLDER)
    # for i in range(average_spec_diff.shape[0]):
    #     writer.add_scalar('spectrogram_diff', average_spec_diff[i], i)
        # print(average_spec_diff.shape)
        # plt.hist(average_spec_diff.cpu().numpy())
        # plt.savefig('fig.png', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune HiFi-GAN.")
    parser.add_argument(
        "--dataset_dir",
        default='/mntcephfs/lab_data/lijiaqi/adver_out/dataset_iphone/*.wav',
        # default='/mntcephfs/lab_data/lijiaqi/ASGSR/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10/*.wav',
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--exp_wavs_dir",
        default=str('/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/recording_dataset_iphone/*.wav'),
        # default=str('/mntcephfs/lab_data/lijiaqi/ASGSR/device_recordings/recovered_pulse/iphone_pgd_xvec_k40s_0.5m/*.wav'),
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--resume",
        default='/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0824_mseloss/model-best.pt',
        help="path to the checkpoint to resume from",
        type=Path,
    )
    parser.add_argument(
        "--finetune",
        default=False,
        help="whether to finetune (note that a resume path must be given)",
        action="store_true",
    )
    args = parser.parse_args()
    eval_spec(args)