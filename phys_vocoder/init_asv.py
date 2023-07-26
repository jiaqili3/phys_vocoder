from easydict import EasyDict as edict
import sys
import torch
import glob

config = edict()

# ---------------------------------------- ASV model ---------------------------------------- #
sys.path.append('..')
from attack.models.RawNet import RawNet3
from attack.models.ECAPATDNN import ECAPATDNN
from attack.models.ResNetSE34V2 import ResNetSE34V2
from attack.models.tdnn import XVEC, XVEC1
from attack.models.model_config import config as model_config

# config.models = [ResNetSE34V2, RawNet3, ECAPATDNN, XVEC1]
config.model = ECAPATDNN

if config.model == RawNet3:
    config.model = RawNet3(**model_config['RawNet3'])
    config.model.load_state_dict(torch.load('../pretrained_models/rawnet3.pt'))
    config.model.threshold = 0.3295809328556061
elif config.model == ECAPATDNN:
    config.model = ECAPATDNN(**model_config['ECAPATDNN'])
    config.model.load_state_dict(torch.load('../pretrained_models/ECAPATDNN.pth'))
    config.model.threshold = 0.33709782361984253
elif config.model == ResNetSE34V2:
    config.model = ResNetSE34V2(**model_config['ResNetSE34V2'])
    config.model.load_state_dict(torch.load('../pretrained_models/ResNetSE34V2.pth'))
    config.model.threshold = 0.3702884316444397
elif config.model == XVEC:
    config.model = XVEC(**model_config['XVEC'])
    config.model.load_state_dict(torch.load('../pretrained_models/XVEC.pth'))
    config.model.threshold = 0.879676103591919
elif config.model == XVEC1:
    config.model = XVEC1()
    config.model.load_state_dict(torch.load('../pretrained_models/XVEC1.pth'))
    config.model.threshold = 0.28246

