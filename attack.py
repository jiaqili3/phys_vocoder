from initializer import config
import torch
import torchaudio
from torch.utils.data import DataLoader
import os

from attack.attacks.pgd import PGD

from dataset.asvspoof2019 import ASVspoof2019

device = 'cuda:0'

model = config.model
model = model.to(device)
phys_vocoder = config.phys_vocoder_model
phys_vocoder = phys_vocoder.to(device)

adver_dir = '/mnt/workspace/lijiaqi/phys_vocoder/adver_out'

class CombinedModel(torch.nn.Module):
    def __init__(self, model, phys_vocoder):
        super(CombinedModel, self).__init__()
        self.model = model
        self.phys_vocoder = phys_vocoder
    def forward(self, x1, x2):
        # x1 enroll, x2 test
        # return (decisions, cos sim)
        x2 = self.phys_vocoder(x2)
        return self.model.make_decision_sv(x1,x2)

combined_model = CombinedModel(model, phys_vocoder)

attacker = PGD(combined_model)
attacker = attacker.to(device)

# data
dataset_name = 'ASVspoof2019'
dataset_config = config.data[dataset_name].dataset
dataset = ASVspoof2019(**dataset_config)
dataloader = DataLoader(dataset, num_workers=4, batch_size=1)
print('dataset: {}'.format(dataset_name))
print('dataset size: {}'.format(len(dataset)))

def save_audio(advers, spk1_file_ids, spk2_file_ids, root, success, fs=16000):
    result_file_path = os.path.join(root, 'attackResult.txt')
    result_file = open(result_file_path, mode='a+')
    for adver, spk1_file_id, spk2_file_id, suc in zip(advers[:, :], spk1_file_ids, spk2_file_ids, success):
        file_name = '{}_{}'.format(spk1_file_id, spk2_file_id)
        adver_path = os.path.join(root, file_name + ".wav")
        adver = adver.detach().cpu()
        adver = torch.unsqueeze(adver, 0)
        torchaudio.save(adver_path, adver, fs)
        result_file.write('{} {}\n'.format(file_name, suc))
    result_file.close()

# attack
success_cnt = 0
total_cnt = 0

for item in dataloader:
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
    # speaker file id
    spk1_file_ids = item[config.data[dataset_name].enroll_file_index]
    spk2_file_ids = item[config.data[dataset_name].test_file_index]

    flag = False
    for spk1_file_id, spk2_file_id in zip(spk1_file_ids, spk2_file_ids):
        file_name = '{}_{}'.format(spk1_file_id, spk2_file_id)
        adver_path = os.path.join(adver_dir, file_name + ".wav")
        if os.path.exists(adver_path):
            print('adversarial audio already exists: {}'.format(adver_path))
            flag = True
            break
    if flag:
        continue

    adver, success = attacker(x1, x2, y)

    if len(adver.size()) == 3:
        adver = adver.squeeze(1)
    save_audio(advers=adver, spk1_file_ids=spk1_file_ids, spk2_file_ids=spk2_file_ids, root=adver_dir,success=success)
    success_cnt += sum(success)
    print('success rate: {}/{}={}'.format(success_cnt, total_cnt, success_cnt / total_cnt))
