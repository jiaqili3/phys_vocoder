import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio.transforms as transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils.utils import compute_PESQ
import pdb

import sys
import os
sys.path.append('..')
sys.path.append('.')
from phys_vocoder.dataset import WavDataset
# from hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram

from phys_vocoder.unet.unet import UNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# BATCH_SIZE = 256
BATCH_SIZE = 256
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
VALIDATION_INTERVAL = 1000
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 5000

def spec_loss(out, tgt, spectrogram):
    spec_out = spectrogram(out)
    spec_tgt = spectrogram(tgt)
    return F.l1_loss(spec_out, spec_tgt)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.tight_layout()
    return fig

def train_model(rank, world_size, args):
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://localhost:54328",
    )
    spectrogram = transforms.Spectrogram(
                n_fft=1024,
                win_length=1024,
                hop_length=160,
                onesided=True,
            ).to(rank)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    generator = UNet().to(rank)

    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=LEARNING_RATE_DECAY
    )
    train_dataset = WavDataset(
        ori_wavs_dir=args.dataset_dir,
        exp_wavs_dir=args.exp_wavs_dir,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=True,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

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
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location={"cuda:0": f"cuda:{rank}"})
        generator.load_state_dict(checkpoint["generator"]["model"])
        optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        global_step, best_loss = checkpoint["step"], checkpoint["loss"]
    else:
        global_step, best_loss = 0, float("inf")

    generator = DDP(generator, device_ids=[rank])


    # if args.finetune:
    #     global_step, best_loss = 0, float("inf")

    n_epochs = EPOCHS
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        generator.train()
        average_loss_generator = 0
        for i, (_, src_wav, tgt_wav) in enumerate(train_loader, 1):
            src_wav, tgt_wav = src_wav.to(rank), tgt_wav.to(rank)

            out = generator(src_wav)

            # Generator
            optimizer_generator.zero_grad()

            # loss_generator = F.l1_loss(out, tgt_wav)
            # change loss
            loss_generator = spec_loss(out, tgt_wav, spectrogram)
            loss_generator.backward()
            optimizer_generator.step()

            global_step += 1 * world_size * BATCH_SIZE

            average_loss_generator += (
                loss_generator.item() - average_loss_generator
            ) / i

            if rank == 0:
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "train/loss_generator",
                        loss_generator.item(),
                        global_step,
                    )

            if global_step % VALIDATION_INTERVAL == 0:
            # if True:
                generator.eval()

                average_validation_loss = 0
                average_pesq = 0
                for j, (_, src, tgt) in enumerate(validation_loader, 1):
                    src, tgt = src.to(rank), tgt.to(rank)

                    with torch.no_grad():
                        out = generator(src)

                    # validation_loss = F.l1_loss(out, tgt)
                    # change loss
                    validation_loss = spec_loss(out, tgt, spectrogram)
                    pesq = compute_PESQ(out.reshape(-1).cpu().numpy(), tgt.reshape(-1).cpu().numpy())
                    average_validation_loss += (
                        validation_loss.item() - average_validation_loss
                    ) / j
                    average_pesq += (
                        pesq - average_pesq
                    ) / j


                    if rank == 0:
                        if j <= NUM_GENERATED_EXAMPLES:
                            writer.add_audio(
                                f"src/wav_{j}",
                                src,
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_audio(
                                f"out/wav_{j}",
                                out,
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_audio(
                                f"tgt/wav_{j}",
                                tgt,
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_audio(
                                f"wav_diff/wav_{j}",
                                src-tgt,
                                global_step,
                                sample_rate=16000,
                            )
                            spec_diff = torchaudio.transforms.Spectrogram(n_fft=128).to(rank)(src-tgt)
                            spec_diff = spec_diff.squeeze(0).squeeze(0).cpu()

                            writer.add_figure(
                                f"spec_diff/spec_{j}",
                                plot_spectrogram(spec_diff.t().numpy()),
                                global_step,
                            )
                            spec_diff = torch.sum(spec_diff.t(), dim=0)
                            for i in range(spec_diff.size()[0]):
                                writer.add_scalar(
                                    f"spec_diff/bin_{i}",
                                    spec_diff[i],
                                    global_step,
                                )
                            


                generator.train()

                if rank == 0:
                    writer.add_scalar(
                        "validation/loss", average_validation_loss, global_step
                    )
                    writer.add_scalar(
                        "validation/pesq", average_pesq, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, global step: {global_step}, loss: {average_validation_loss:.4f}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = average_validation_loss

                    if rank == 0:
                        state = {
                            "generator": {
                                "model": generator.module.state_dict(),
                                "optimizer": optimizer_generator.state_dict(),
                                "scheduler": scheduler_generator.state_dict(),
                            },
                            "step": global_step,
                            "loss": average_validation_loss,
                        }
                        torch.save(state, args.checkpoint_dir / f"model-{global_step}.pt")

        scheduler_generator.step()

        logger.info(
            f"train -- epoch: {epoch}, generator loss: {average_loss_generator:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune HiFi-GAN.")
    parser.add_argument(
        "--dataset_dir",
        default='/mntcephfs/lab_data/wangli/concat/*.wav',
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--exp_wavs_dir",
        default=str('/mntcephfs/lab_data/lijiaqi/ASGSR/device_recordings/recovered/iphone_concat_k40s_0.5m/*.wav'),
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_dir",
        default='/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0712_specloss',
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        default=None,
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

    # display training setup info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")

    # clear handlers
    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
    # train_model(0,1,args)
