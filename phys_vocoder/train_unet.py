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
import random
random.seed(16)
import glob

from init_asv import config

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
FINETUNE_LEARNING_RATE = 1e-6
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.995
WEIGHT_DECAY = 1e-5
EPOCHS = 200
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 20000 // BATCH_SIZE * BATCH_SIZE
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 40000 // BATCH_SIZE * BATCH_SIZE

def loss_func(out, tgt, spectrogram, enroll, asv_model):
    spec_out = spectrogram(out)
    spec_tgt = spectrogram(tgt)
    spec_loss = F.l1_loss(spec_out, spec_tgt)
    # wav_loss = F.l1_loss(out, tgt)
    return spec_loss + 100*F.mse_loss(out, tgt)
    # return spec_loss

    enroll = torch.cat([enroll.unsqueeze(0)]*spec_out.shape[0], dim=0)
    _, cos1 = asv_model.make_decision_SV(enroll, out)
    _, cos2 = asv_model.make_decision_SV(enroll, tgt)
    # print(torch.abs(cos1-cos2).mean())
    # print(0.1*spec_loss + 0.9*wav_loss)
    # pdb.set_trace()
    return 0.1*spec_loss + 0.9*wav_loss + torch.abs(cos1-cos2).mean()

    # return 0.1*spec_loss + 0.9*wav_loss + 6*torch.abs(cos1-cos2).mean()
    # return spec_loss + 50*torch.abs(cos1-cos2).mean()
    # return torch.abs(cos1-cos2).mean()

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
        init_method="tcp://localhost:54307",
    )
    if config.model is not None:
        asv_model = config.model.to(rank)
        asv_model.eval()
    else:
        asv_model = None
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80).to(rank)


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
        batch_size=32,
        shuffle=False,
        num_workers=4,
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


    if args.finetune:
        global_step, best_loss = 0, float("inf")

    n_epochs = EPOCHS
    start_epoch = global_step // BATCH_SIZE // world_size // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")
    enroll_flist = glob.glob('./enroll_waveforms/*.wav')

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        generator.train()
        average_loss_generator = 0
        for i, (_, src_wav, tgt_wav) in enumerate(train_loader, 1):
            src_wav, tgt_wav = src_wav.to(rank), tgt_wav.to(rank)
            out = generator(src_wav)

            enroll_waveform = random.choice(enroll_flist)
            enroll_waveform, _ = torchaudio.load(enroll_waveform)
            enroll_waveform = enroll_waveform.to(rank)

            # Generator
            optimizer_generator.zero_grad()

            # loss_generator = F.l1_loss(out, tgt_wav)
            # change loss
            loss_generator = loss_func(out, tgt_wav, spectrogram, enroll_waveform, asv_model)
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
                num_generated = 0
                for j, (_, src, tgt) in enumerate(validation_loader, 1):
                    src, tgt = src.to(rank), tgt.to(rank)

                    enroll_waveform = random.choice(enroll_flist)
                    enroll_waveform, _ = torchaudio.load(enroll_waveform)
                    enroll_waveform = enroll_waveform.to(rank)

                    with torch.no_grad():
                        out = generator(src)

                    # pdb.set_trace()
                    # validation_loss = F.l1_loss(out, tgt)
                    # change loss
                    validation_loss = loss_func(out, tgt, spectrogram, enroll_waveform, asv_model)
                    try:
                        pesq = 0.0
                        for i in range(out.shape[0]):
                            pesq = pesq + compute_PESQ(out[i].reshape(-1).cpu().numpy(), tgt[i].reshape(-1).cpu().numpy())
                            # pesq = compute_PESQ(out.reshape(-1).cpu().numpy(), tgt.reshape(-1).cpu().numpy())
                        pesq = pesq / out.shape[0]
                    except:
                        print(out.shape, tgt.shape)
                        pesq = 0.0

                    average_validation_loss += (
                        validation_loss.item() - average_validation_loss
                    ) / j
                    average_pesq += (
                        pesq - average_pesq
                    ) / j

                    # spec_diff1 = spec_out - spec_tgt
                    # spec_diff1 = spec_diff.squeeze(0).squeeze(0)
                    # shape: (n_bin, time)

                    # if average_spec_diff is None:
                    #     average_spec_diff = torch.sum(spec_diff, dim=1)
                    # else:
                    #     average_spec_diff += (
                    #         torch.sum(spec_diff, dim=1) - average_spec_diff
                    #     ) / j
                    if rank == 0:
                        spec_out_ = spectrogram(out)
                        spec_tgt_ = spectrogram(tgt)
                        spec_src_ = spectrogram(src)

                        for n in range(spec_out_.shape[0]):
                            # writer.add_audio(
                            #     f"src/wav_{n}",
                            #     src[n],
                            #     0,
                            #     sample_rate=16000,
                            # )
                            # writer.add_audio(
                            #     f"out/wav_{n}",
                            #     out[n],
                            #     0,
                            #     sample_rate=16000,
                            # )
                            # writer.add_audio(
                            #     f"tgt/wav_{n}",
                            #     tgt[n],
                            #     0,
                            #     sample_rate=16000,
                            # )
                            # spec_out = spec_out_[n].squeeze(0).cpu()
                            # spec_tgt = spec_tgt_[n].squeeze(0).cpu()
                            # spec_src = spec_src_[n].squeeze(0).cpu()
                            # spec_outminsrc = spec_out - spec_src
                            # spec_tgtminsrc = spec_tgt - spec_src
                            # writer.add_figure(
                            #     f"spec_diff/spec_{j}",
                            #     plot_spectrogram(spec_diff.cpu().numpy()),
                            #     global_step,
                            # )
                            # writer.add_figure(
                            #     f"spec_out/spec_{j}",
                            #     plot_spectrogram(spec_out.numpy()),
                            #     0,
                            # )
                            # writer.add_figure(
                            #     f"spec_tgt/spec_{j}",
                            #     plot_spectrogram(spec_tgt.numpy()),
                            #     0,
                            # )
                            # writer.add_figure(
                            #     f"spec_outminsrc/spec_{j}",
                            #     plot_spectrogram(spec_outminsrc.numpy()),
                            #     0,
                            # )
                            # writer.add_figure(
                            #     f"spec_tgtminsrc/spec_{j}",
                            #     plot_spectrogram(spec_tgtminsrc.numpy()),
                            #     0,
                            # )
                            # writer.add_figure(
                            #     f"spec_outmintgt/spec_{j}",
                            #     plot_spectrogram(spec_out.numpy() - spec_tgt.numpy()),
                            #     0,
                            # )
                                
                            num_generated += 1
                            if num_generated > NUM_GENERATED_EXAMPLES:
                                break


                generator.train()

                if rank == 0:
                    # for i in range(average_spec_diff.shape[0]):
                    #     writer.add_scalar(
                    #         f"avg_spec_diff/bin_{i}",
                    #         average_spec_diff[i],
                    #         global_step,
                    #     )
                    # plot the avg diff spectrogram
                    # writer.add_figure(
                    #     f"avg_diff/step {global_step}",
                    #     plot_spectrogram(average_spec_diff*torch.ones(average_spec_diff.shape[0])),
                    #     global_step,
                    # )

                    writer.add_scalar(
                        "validation/loss", average_validation_loss, global_step
                    )
                    writer.add_scalar(
                        "validation/pesq", average_pesq, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, global step: {global_step}, loss: {average_validation_loss:.4f}, pesq: {average_pesq}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:

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
                    if new_best:
                        state = {
                            "generator": {
                                "model": generator.module.state_dict(),
                                "optimizer": optimizer_generator.state_dict(),
                                "scheduler": scheduler_generator.state_dict(),
                            },
                            "step": global_step,
                            "loss": average_validation_loss,
                        }
                        logger.info("-------- new best model found!")
                        best_loss = average_validation_loss
                        torch.save(state, args.checkpoint_dir / f"model-best.pt")

        scheduler_generator.step()

        logger.info(
            f"train -- epoch: {epoch}, generator loss: {average_loss_generator:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune UNet.")
    parser.add_argument(
        "--dataset_dir",
        default='/mntcephfs/lab_data/lijiaqi/adver_out/dataset_iphone/*.wav',
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--exp_wavs_dir",
        default=str('/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/recording_dataset_iphone/*.wav'),
        metavar="dataset-dir",
        help="path to the preprocessed data directory",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_dir",
        default='/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0824_melandl2wavloss',
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
