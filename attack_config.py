from easydict import EasyDict as edict
import sys

sys.path.append('phys_vocoder')
config = edict()


# ---------------------------------------- physical vocoder ---------------------------------------- #
# hifigan
config.use_physical_vocoder = True
from hifigan.generator import HifiganEndToEnd, HifiganGenerator
config.phys_vocoder_model = HifiganEndToEnd


# ---------------------------------------- Attack ---------------------------------------
config.attack = edict()
config.attack.attack_name = 'FAKEBOB'
# PGD
config.attack.PGD = edict()
config.attack.PGD.task = 'SV'
config.attack.PGD.targeted = False
config.attack.PGD.step_size = 0.0004
config.attack.PGD.epsilon = 0.005
config.attack.PGD.max_iter = 10
config.attack.PGD.batch_size = 1
config.attack.PGD.num_random_init = 0
config.attack.PGD.EOT_size = 1
config.attack.PGD.EOT_batch_size = 1
config.attack.PGD.verbose = 1

# FAKEBOB
config.attack.FAKEBOB = edict()
config.attack.FAKEBOB.threshold_estimated = None
config.attack.FAKEBOB.task = 'SV'
# config.attack.FAKEBOB.targeted = False
config.attack.FAKEBOB.confidence = 0
config.attack.FAKEBOB.epsilon = 0.005
config.attack.FAKEBOB.max_iter = 1000
config.attack.FAKEBOB.max_lr = 0.001
config.attack.FAKEBOB.min_lr = 1e-6
config.attack.FAKEBOB.samples_per_draw = 16
config.attack.FAKEBOB.samples_per_draw_batch_size = 16
config.attack.FAKEBOB.sigma = 0.001
config.attack.FAKEBOB.momentum = 0.9
config.attack.FAKEBOB.plateau_length = 5
config.attack.FAKEBOB.plateau_drop = 2.0
config.attack.FAKEBOB.stop_early = True
config.attack.FAKEBOB.stop_early_iter = 100
config.attack.FAKEBOB.batch_size = 1
config.attack.FAKEBOB.EOT_size = 1
config.attack.FAKEBOB.EOT_batch_size = 1
config.attack.FAKEBOB.thresh_est_wav_path = [
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0005009.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0004411.flac'],
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003405.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003825.flac'],
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0000210.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003636.flac']
]

config.attack.FAKEBOB.verbose = 1
config.attack.FAKEBOB.thresh_est_step = 0.1  # the smaller, the accurate, but the slower

# ---------------------------------------- dataset --------------------------------------- #
config.data = edict()
config.data.dataset_name = 'ASVspoof2019'

# VoxCeleb1Verification
config.data.VoxCeleb1Verification = edict()
config.data.VoxCeleb1Verification.dataset = edict()
config.data.VoxCeleb1Verification.dataloader = edict()
config.data.VoxCeleb1Verification.dataset.root = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1'
config.data.VoxCeleb1Verification.dataset.meta_list_path = '/mntnfs/lee_data1/wangli/ASGSR/Vox1_10/veri_test_10_0.txt'
config.data.VoxCeleb1Verification.dataloader.batch_size = 1
config.data.VoxCeleb1Verification.waveform_index_spk1 = 0
config.data.VoxCeleb1Verification.waveform_index_spk2 = 1
config.data.VoxCeleb1Verification.label_index = 3
config.data.VoxCeleb1Verification.enroll_file_index = 4
config.data.VoxCeleb1Verification.test_file_index = 5  # spk1 for enroll, spk2 for test

config.data.ASVspoof2019 = edict()
config.data.ASVspoof2019.dataset = edict()
config.data.ASVspoof2019.dataloader = edict()
config.data.ASVspoof2019.dataset.data_file = '/home/wangli/ASGSR/audio_record/enroll_eval_pairs.txt'
config.data.ASVspoof2019.dataset.train_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'
config.data.ASVspoof2019.dataset.dev_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'
config.data.ASVspoof2019.dataset.eval_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'
config.data.ASVspoof2019.dataloader.batch_size = 1
config.data.ASVspoof2019.waveform_index_spk1 = 0
config.data.ASVspoof2019.waveform_index_spk2 = 1
config.data.ASVspoof2019.label_index = 3
config.data.ASVspoof2019.enroll_file_index = 4
config.data.ASVspoof2019.test_file_index = 5  # spk1 for enroll, spk2 for test

# ---------------------------------------- ASV model ---------------------------------------- #
config.model = edict()
config.model.model_name = 'RawNet3'

# ResNetSE34V2
config.model.ResNetSE34V2 = edict()
config.model.ResNetSE34V2.save_path = '/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.pth'
config.model.ResNetSE34V2.threshold = 0.3702884316444397

# ECAPATDNN
config.model.ECAPATDNN = edict()
config.model.ECAPATDNN.save_path = '/home/wangli/ASGSR/pretrained_models/ECAPATDNN/ECAPATDNN.pth'
config.model.ECAPATDNN.threshold = 0.33709782361984253

# RawNet3
config.model.RawNet3 = edict()
config.model.RawNet3.save_path = '/home/wangli/ASGSR/pretrained_models/RawNet3/model.pt'
config.model.RawNet3.threshold = 0.3295809328556061

# XVEC
config.model.XVEC = edict()
config.model.XVEC.save_path = '/home/wangli/ASGSR/SR/exps/XVEC/20230321145333/model_epoch40_ValidAcc0.8902085747392816.pth'
config.model.XVEC.threshold = 0.879676103591919

# HifiGAN
config.model.hifigan = edict()
config.model.hifigan.save_path = ''

