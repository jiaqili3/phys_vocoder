from easydict import EasyDict as edict
import sys
from attack.attacks.pgd import PGD
import torch


config = edict()


# ---------------------------------------- physical vocoder ---------------------------------------- #
from phys_vocoder.hifigan.generator import HifiganEndToEnd
from phys_vocoder.unet.unet import UNetEndToEnd

config.phys_vocoder_model = UNetEndToEnd

# HifiGAN
if config.phys_vocoder_model == HifiganEndToEnd:
    config.phys_vocoder_model = HifiganEndToEnd()
    config.phys_vocoder_model.load_model('/mntcephfs/lab_data/lijiaqi/hifigan-checkpoints/0630/model-115000.pt')

# unet
elif config.phys_vocoder_model == UNetEndToEnd:
    config.phys_vocoder_model = UNetEndToEnd()
    config.phys_vocoder_model.load_model('/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0702/model-45000.pt')

# ---------------------------------------- ASV model ---------------------------------------- #
from attack.models.RawNet import RawNet3
from attack.models.ECAPATDNN import ECAPATDNN
from attack.models.ResNetSE34V2 import ResNetSE34V2
from attack.models.tdnn import XVEC, XVEC1
from attack.models.model_config import config as model_config

config.model = XVEC1

if config.model == RawNet3:
    config.model = RawNet3(**model_config['RawNet3'])
    config.model.load_state_dict(torch.load('./pretrained_models/rawnet3.pt'))
    config.model.threshold = 0.3295809328556061
elif config.model == ECAPATDNN:
    config.model = ECAPATDNN(**model_config['ECAPATDNN'])
    config.model.load_state_dict(torch.load('./pretrained_models/ECAPATDNN.pth'))
    config.model.threshold = 0.33709782361984253
elif config.model == ResNetSE34V2:
    config.model = ResNetSE34V2(**model_config['ResNetSE34V2'])
    config.model.load_state_dict(torch.load('./pretrained_models/ResNetSE34V2.pth'))
    config.model.threshold = 0.3702884316444397
elif config.model == XVEC:
    config.model = XVEC(**model_config['XVEC'])
    config.model.load_state_dict(torch.load('./pretrained_models/XVEC.pth'))
    config.model.threshold = 0.879676103591919
elif config.model == XVEC1:
    # XVEC1
    config.model = XVEC1()
    config.model.load_state_dict(torch.load('./pretrained_models/XVEC1.model'))
    config.model.threshold = 0.28246

# ResNetSE34V2
# config.model.ResNetSE34V2 = edict()
# config.model.ResNetSE34V2.save_path = '/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.pth'
# config.model.ResNetSE34V2.threshold = 0.3702884316444397

# # ECAPATDNN
# config.model.ECAPATDNN = edict()
# config.model.ECAPATDNN.save_path = '/home/wangli/ASGSR/pretrained_models/ECAPATDNN/ECAPATDNN.pth'
# config.model.ECAPATDNN.threshold = 0.33709782361984253

# RawNet3
    # config.model.save_path = '/home/wangli/ASGSR/pretrained_models/RawNet3/model.pt'

# XVEC
# config.model.XVEC = edict()
# config.model.XVEC.save_path = '/home/wangli/ASGSR/SR/exps/XVEC/20230321145333/model_epoch40_ValidAcc0.8902085747392816.pth'
# config.model.XVEC.threshold = 0.879676103591919


# ---------------------------------------- Attack ---------------------------------------

config.attack = edict()
config.attack.adv_dir = '/mntcephfs/lab_data/lijiaqi/adver_out/'

config.attack.steps = 300
# config.attack.steps = 10
config.attack.alpha = 0.0004
# config.attack.eps = 0.005
config.attack.eps = 0.02


# config.attack.attack_class = PGD

# # PGD
# if config.attack.attack_class == PGD:
#     kwargs = {
#         'model': None,
#         'task': 'SV',
#         'epsilon': 0.005,
#         'step_size': 0.0004,
#         'max_iter': 10,
#         'num_random_init': 0,
#         'targeted': False,
#         'batch_size': 1,
#         'EOT_size': 1,
#         'EOT_batch_size': 1,
#         'verbose': 1,
#     }
#     config.attack.attack_class = PGD(**kwargs)

# ---------------------------------------- dataset --------------------------------------- #
config.data = edict()
config.data.dataset_name = 'ASVspoof2019'

# VoxCeleb1Verification
# config.data.VoxCeleb1Verification = edict()
# config.data.VoxCeleb1Verification.dataset = edict()
# config.data.VoxCeleb1Verification.dataloader = edict()
# config.data.VoxCeleb1Verification.dataset.root = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1'
# config.data.VoxCeleb1Verification.dataset.meta_list_path = '/mntnfs/lee_data1/wangli/ASGSR/Vox1_10/veri_test_10_0.txt'
# config.data.VoxCeleb1Verification.dataloader.batch_size = 1
# config.data.VoxCeleb1Verification.waveform_index_spk1 = 0
# config.data.VoxCeleb1Verification.waveform_index_spk2 = 1
# config.data.VoxCeleb1Verification.label_index = 3
# config.data.VoxCeleb1Verification.enroll_file_index = 4
# config.data.VoxCeleb1Verification.test_file_index = 5  # spk1 for enroll, spk2 for test

config.data.ASVspoof2019 = edict()
config.data.ASVspoof2019.dataset = edict()
config.data.ASVspoof2019.dataloader = edict()
config.data.ASVspoof2019.dataset.data_file = './dataset/enroll_eval_pairs.txt'
# config
config.data.ASVspoof2019.dataset.train_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'
config.data.ASVspoof2019.dataset.dev_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'
config.data.ASVspoof2019.dataset.eval_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'

config.data.ASVspoof2019.dataloader.batch_size = 1
config.data.ASVspoof2019.waveform_index_spk1 = 0
config.data.ASVspoof2019.waveform_index_spk2 = 1
config.data.ASVspoof2019.label_index = 3
config.data.ASVspoof2019.enroll_file_index = 4
config.data.ASVspoof2019.test_file_index = 5  # spk1 for enroll, spk2 for test

