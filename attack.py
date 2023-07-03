from init_attack import config
import torch
import torchaudio
from torch.utils.data import DataLoader
import os

from attack.attacks.pgd import PGD
import logging
from dataset.asvspoof2019 import ASVspoof2019

device = 'cuda:0'

model = config.model
model = model.to(device)
phys_vocoder = config.phys_vocoder_model
if phys_vocoder is not None:
    phys_vocoder = phys_vocoder.to(device)


adver_dir = config.attack.adv_dir
adver_dir = os.path.join(adver_dir, f'{model.__class__.__name__}_{config.phys_vocoder_model.__class__.__name__}_{config.attack.steps}_{config.attack.alpha}_{config.attack.eps}')
os.makedirs(adver_dir, exist_ok=True)
logging.basicConfig(filename=f'{adver_dir}/log', encoding='utf-8', level=logging.DEBUG, force=True)
print(f'adv samples saved to {adver_dir}')
logging.info(config.attack)

class CombinedModel(torch.nn.Module):
    def __init__(self, model, phys_vocoder):
        super(CombinedModel, self).__init__()
        self.phys_vocoder = phys_vocoder
        self.model = model
        if self.phys_vocoder is not None:
            logging.info('using physical vocoder')
    def forward(self, x1, x2):
        # x1 enroll, x2 test
        # return (decisions, cos sim)
        if self.phys_vocoder is not None:
            x2_voc = self.phys_vocoder(x2)
        else:
            x2_voc = x2
        return self.model.make_decision_SV(x1, x2_voc)

model.eval()
combined_model = CombinedModel(model, phys_vocoder)
combined_model.eval()
combined_model.threshold = model.threshold

attacker = PGD(combined_model, steps=config.attack.steps, alpha=config.attack.alpha, random_start=False, eps=config.attack.eps)

# data
dataset_name = 'ASVspoof2019'
dataset_config = config.data[dataset_name].dataset
dataset = ASVspoof2019(**dataset_config)
dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)
logging.info('dataset: {}'.format(dataset_name))
logging.info('dataset size: {}'.format(len(dataset)))

def save_audio(advers, spk1_file_ids, spk2_file_ids, root, success, fs=16000):
    result_file_path = os.path.join(root, 'attackResult.txt')
    result_file = open(result_file_path, mode='a+')
    for adver, spk1_file_id, spk2_file_id, suc in zip(advers[:, :], spk1_file_ids, spk2_file_ids, success):
        file_name = '{}_{}'.format(spk1_file_id, spk2_file_id)
        adver_path = os.path.join(root, file_name + ".wav")
        adver = adver.detach().cpu()
        adver = torch.unsqueeze(adver, 0)
        print(f'saved to {adver_path}')
        torchaudio.save(adver_path, adver, fs)
        result_file.write('{} {}\n'.format(file_name, suc))
    result_file.close()

# attack
success_cnt = 0
total_cnt = 0


for runno, item in enumerate(dataloader):
    # shape: (batch_size, channels, audio_len)
    x1 = item[config.data[dataset_name].waveform_index_spk1]
    # to extract single channel?
    x2 = item[config.data[dataset_name].waveform_index_spk2]
    y = item[config.data[dataset_name].label_index]

    # target attack & untarget attack
    if int(y) == 1:
        # only pick bona-fide audio
        continue
    total_cnt += 1
    x1, x2, y = x1.to(device), x2.to(device), y.to(device)


    # decision, score = model.make_decision_SV(x1, x2)
    # logging.info(score)
    # continue



    # speaker file id
    spk1_file_ids = item[config.data[dataset_name].enroll_file_index]
    spk2_file_ids = item[config.data[dataset_name].test_file_index]

    flag = False
    for spk1_file_id, spk2_file_id in zip(spk1_file_ids, spk2_file_ids):
        file_name = '{}_{}'.format(spk1_file_id, spk2_file_id)
        adver_path = os.path.join(adver_dir, file_name + ".wav")
        # skip already existing ones
        if os.path.exists(adver_path):
            logging.info('adversarial audio already exists: {}'.format(adver_path))
            flag = True
            break
    if flag:
        continue

    adver, success = attacker(x1, x2, y, runno)

    if len(adver.size()) == 3:
        adver = adver.squeeze(1)
    save_audio(advers=adver, spk1_file_ids=spk1_file_ids, spk2_file_ids=spk2_file_ids, root=adver_dir,success=success)
    success_cnt += sum(success)
    print('success rate: {}/{}={}'.format(success_cnt, total_cnt, success_cnt / total_cnt))
