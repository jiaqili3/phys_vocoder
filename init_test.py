from easydict import EasyDict as edict
import sys
import torch


config = edict()

config.adv_dirs = [
    '/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/iphone_RawNet3_UNetEndToEnd_300_0.0004_0.02',
    ]

# ---------------------------------------- ASV model ---------------------------------------- #
from attack.models.RawNet import RawNet3
from attack.models.ECAPATDNN import ECAPATDNN
from attack.models.ResNetSE34V2 import ResNetSE34V2
from attack.models.tdnn import XVEC, XVEC1
from attack.models.model_config import config as model_config

# phys vocoder
from phys_vocoder.hifigan.generator import HifiganEndToEnd
from phys_vocoder.unet.unet import UNetEndToEnd

config.phys_vocoder_model = UNetEndToEnd

# config.models = [ResNetSE34V2, RawNet3, ECAPATDNN, XVEC]
config.model = RawNet3
config.use_phys_vocoder = False

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
    config.model = XVEC(**model_config['XVEC'])
    config.model.load_state_dict(torch.load('./pretrained_models/XVEC1.pth'))
    config.model.threshold = 0.28246

class CombinedModel(torch.nn.Module):
    def __init__(self, model, phys_vocoder):
        super(CombinedModel, self).__init__()
        self.phys_vocoder = phys_vocoder
        self.model = model
        self.__class__.__name__ = f'{phys_vocoder.__class__.__name__}_{model.__class__.__name__}'
    def forward(self, x1, x2):
        # x1 enroll, x2 test
        # return (decisions, cos sim)
        if self.phys_vocoder is not None:
            x2_voc = self.phys_vocoder(x2)
        else:
            x2_voc = x2
        return self.model.make_decision_SV(x1, x2_voc)
    def make_decision_SV(self, x1, x2):
        return self.forward(x1, x2)
# HifiGAN
if config.phys_vocoder_model == HifiganEndToEnd:
    config.phys_vocoder_model = HifiganEndToEnd()
    config.phys_vocoder_model.load_model('/mntcephfs/lab_data/lijiaqi/hifigan-checkpoints/0630/model-115000.pt')

# unet
elif config.phys_vocoder_model == UNetEndToEnd:
    config.phys_vocoder_model = UNetEndToEnd()
    config.phys_vocoder_model.load_model('/mntcephfs/lab_data/lijiaqi/unet_checkpoints/0702/model-45000.pt')

if config.use_phys_vocoder:
    thres = config.model.threshold
    config.model = CombinedModel(config.model, config.phys_vocoder_model)
    config.model.eval()
    config.model.threshold = thres


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

