# test the attack accuracy against ASV models

from init_test import config
import torch
import torchaudio
from torch.utils.data import DataLoader
import os
import glob

from dataset.asvspoof2019 import ASVspoof2019

import logging
import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = 'cuda:0'

model = config.model
model = model.to(device)


model.eval()

def transferAttack(device, model, dataloader, out_file, logger):
    num_success = 0.0
    num_total = 0.0
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader):
            enroll_waveforms, test_waveforms = item[0], item[1]
            enroll_waveforms = enroll_waveforms.to(device)
            test_waveforms = test_waveforms.to(device)

            enroll_file = item[2]
            test_file = item[3]
            # print(f'enroll_file: {enroll_file} test_file: {test_file} enroll shape: {enroll_waveforms.shape} test shape: {test_waveforms.shape}')


            decisions, cos = model.make_decision_SV(enroll_waveforms, test_waveforms)
            num_total += enroll_waveforms.size(0)
            num_success += decisions.sum().item()
            logger.info('Enroll file: {}, Test file: {}, Decision: {}, Cos: {}'.format(
                enroll_file, test_file, decisions.item(), cos.item()))
            out_file.write('{} {} {} {}\n'.format(enroll_file[0], test_file[0], decisions.item(), cos.item()))
        attack_acc = num_success / num_total
        logger.info('attck acc: {}/{}={}'.format(num_success, num_total, attack_acc))
    return attack_acc


for adv_dir in config.adv_dirs:
    if adv_dir[-1] == '/':
        adv_dir = adv_dir[:-1]  # not include '/'
    attacker_info = adv_dir.split('/')[-1]  # PGD_XVEC-20230305192215_eps-0.001

    output_dir = f'./transfer_attack_exps/{attacker_info}_versus_{model.__class__.__name__}'
    os.makedirs(output_dir, exist_ok=True)

    # data
    dataset_name = 'ASVspoof2019'
    dataset_config = config.data[dataset_name].dataset
    dataset = ASVspoof2019(**dataset_config)

    attack_result_file = f'{adv_dir}/attackResult.txt'
    dataloader = dataset.get_attack_pairs(attack_result_file, adv_dir)

    out_file_path = os.path.join(output_dir, 'transferAttackResult.txt')
    out_file = open(out_file_path, 'w')
    transferAttack(device, model, dataloader, out_file=out_file, logger=logger)
    out_file.close()