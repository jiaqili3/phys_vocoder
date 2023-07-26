import librosa
from pathlib import Path
import torchaudio
import torch
import glob
from attack.models.ECAPATDNN import ECAPATDNN
from attack.models.model_config import config as model_config
from dataset.asvspoof2019 import ASVspoof2019
from init_test import config
import tqdm
import pdb

model = ECAPATDNN(**model_config['ECAPATDNN'])
model.load_state_dict(torch.load('./pretrained_models/ECAPATDNN.pth'))
model.threshold = 0.33709782361984253
device = 'cuda:0'
model = model.to(device)
model.eval()

# attack_pairs = dict()
# with open('attackResult.txt') as f:
#     for line in f:
#         line = line.strip().split(' ')
#         enroll_fname = line[0][:12] + 'wav'  # id10270-x6uYqmx31kE-00001.wav
#         eval_fname = line[0][13:] + '.wav'  # id10270-x6uYqmx31kE-00003_id10273-8cfyJEV7hP8-00004
#         attack_pairs[eval_fname] = enroll_fname

dataset_config = config.data['ASVspoof2019'].dataset
dataset = ASVspoof2019(**dataset_config)
dataloader = dataset.get_attack_pairs('attackResult.txt', '/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/iphone_ECAPATDNN_None_0_0.0004_0.005')

# examine attack effects 
for audio_no, item in enumerate(dataloader):
    enroll_waveforms, test_waveforms = item[0], item[1]
    enroll_waveforms = enroll_waveforms.to(device)
    test_waveforms = test_waveforms.to(device)

    enroll_file = item[2]
    test_file = item[3]
    # print(f'enroll_file: {enroll_file} test_file: {test_file} enroll shape: {enroll_waveforms.shape} test shape: {test_waveforms.shape}')


    decision1, cos1 = model.make_decision_SV(enroll_waveforms, test_waveforms)
    # print(decision1.item(), cos1.item())
    # num_total += enroll_waveforms.size(0)
    # num_success += decisions.sum().item()

    # unrecorded
    ori_waveform, _ = torchaudio.load(f'/mntcephfs/lab_data/lijiaqi/adver_out/ECAPATDNN_None_0_0.0004_0.005/{enroll_file}_{test_file}.wav')
    ori_waveform = ori_waveform.to(device).unsqueeze(0)
    decision2, cos2 = model.make_decision_SV(enroll_waveforms, ori_waveform)
    print('recording cause degradation', cos1.item() - cos2.item())

    # split into k_bins bins 
    k_bins = 4
    for i in range(k_bins):

        

        length = test_waveforms.size()[-1]
        ori_waveform_copy = ori_waveform.clone()
        ori_waveform_copy[:, :, i * length//k_bins : (i+1) * length//k_bins] = 0
        decision3, cos3 = model.make_decision_SV(enroll_waveforms, ori_waveform_copy)
        print('cos1', cos1.item())
        print('cos2', cos2.item())
        print('cos3', cos3.item())
        # print(f'substituting part {i} cause degradation', cos3.item() - cos2.item())
        print(f'proportion of degradation part {i}', (cos3.item() - cos2.item()) / (cos1.item() - cos2.item()))
        # print(cos3.item() - cos2.item())
        # pdb.set_trace()
        torchaudio.save(f'{audio_no}_part{i}.wav', ori_waveform_copy.cpu().squeeze(0), 16000)
    # exit()
    print(f'--------audio {audio_no}------')
    if audio_no > 20: 
        exit()

print('---------------RECORD----------------')
exit()
# recorded
for item in dataloader:
    enroll_waveforms, test_waveforms = item[0], item[1]
    enroll_waveforms = enroll_waveforms.to(device)
    test_waveforms = test_waveforms.to(device)

    enroll_file = item[2]
    test_file = item[3]
    # print(f'enroll_file: {enroll_file} test_file: {test_file} enroll shape: {enroll_waveforms.shape} test shape: {test_waveforms.shape}')


    decision1, cos1 = model.make_decision_SV(enroll_waveforms, test_waveforms)
    # print(decision1.item(), cos1.item())
    # num_total += enroll_waveforms.size(0)
    # num_success += decisions.sum().item()

    # unrecorded
    ori_waveform, _ = torchaudio.load(f'/mntcephfs/lab_data/lijiaqi/adver_out/ECAPATDNN_None_0_0.0004_0.005/{enroll_file}_{test_file}.wav')
    ori_waveform = ori_waveform.to(device).unsqueeze(0)
    decision2, cos2 = model.make_decision_SV(enroll_waveforms, ori_waveform)
    print('recording cause degradation', cos1.item() - cos2.item())

    # split into k_bins bins 
    k_bins = 20
    for i in range(k_bins):

        

        length = test_waveforms.size()[-1]
        ori_waveform_copy = ori_waveform.clone()
        ori_waveform_copy[:, :, i * length//k_bins : (i+1) * length//k_bins] = test_waveforms[:, :, i * length//k_bins : (i+1) * length//k_bins]
        decision3, cos3 = model.make_decision_SV(enroll_waveforms, ori_waveform_copy)
        print('cos1', cos1.item())
        print('cos2', cos2.item())
        print('cos3', cos3.item())
        print(f'substituting part {i} cause degradation', cos3.item() - cos2.item())
        print(f'proportion of degradation part {i}', (cos3.item() - cos2.item()) / (cos1.item() - cos2.item()))
        # print(cos3.item() - cos2.item())
        # pdb.set_trace()
        torchaudio.save(f'part{i}.wav', ori_waveform_copy.cpu().squeeze(0), 16000)
    exit()
for item in dataloader:
    enroll_waveforms, test_waveforms = item[0], item[1]
    enroll_waveforms = enroll_waveforms.to(device)
    test_waveforms = test_waveforms.to(device)

    enroll_file = item[2]
    test_file = item[3]
    # print(f'enroll_file: {enroll_file} test_file: {test_file} enroll shape: {enroll_waveforms.shape} test shape: {test_waveforms.shape}')


    decision1, cos1 = model.make_decision_SV(enroll_waveforms, test_waveforms)
    # print(decision1.item(), cos1.item())
    # num_total += enroll_waveforms.size(0)
    # num_success += decisions.sum().item()

    # unrecorded
    ori_waveform, _ = torchaudio.load(f'/mntcephfs/lab_data/lijiaqi/adver_out/ECAPATDNN_None_0_0.0004_0.005/{enroll_file}_{test_file}.wav')
    ori_waveform = ori_waveform.to(device).unsqueeze(0)
    decision2, cos2 = model.make_decision_SV(enroll_waveforms, ori_waveform)
    print('recording cause degradation', cos1.item() - cos2.item())

    # split into k_bins bins 
    k_bins = 20
    for i in range(k_bins):

        

        length = enroll_waveforms.size()[-1]
        ori_waveform_copy = ori_waveform.clone()
        ori_waveform_copy[:, :, i * length//k_bins : (i+1) * length//k_bins] = test_waveforms[:, :, i * length//k_bins : (i+1) * length//k_bins]
        decision3, cos3 = model.make_decision_SV(enroll_waveforms, ori_waveform_copy)
        print('cos1', cos1.item())
        print('cos2', cos2.item())
        print('cos3', cos3.item())
        print(f'substituting part {i} cause degradation', cos3.item() - cos2.item())
        print(f'proportion of degradation part {i}', (cos3.item() - cos2.item()) / (cos1.item() - cos2.item()))
        # print(cos3.item() - cos2.item())
        # pdb.set_trace()
        torchaudio.save(f'part{i}.wav', ori_waveform_copy.cpu().squeeze(0), 16000)
    exit()

    print('------------')






# dataset_dir = glob.glob('/mntcephfs/lab_data/lijiaqi/adver_out/ECAPATDNN_None_0_0.0004_0.00/*.wav')
# exp_wavs_dir = glob.glob('/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/iphone_ECAPATDNN_None_0_0.0004_0.005/*.wav')
# assert len(dataset_dir) == len(exp_wavs_dir)

# for path1, path2 in zip(dataset_dir, exp_wavs_dir):
#     input_waveform, _ = torchaudio.load(path1)
#     rec_waveform, _ = torchaudio.load(path2)
#     input_waveform = input_waveform.to(device)
#     rec_waveform = rec_waveform.to(device)

#     # get attack score of input waveform
#     decision, cos = model.make_decision_SV()

#     # k_bins bins, 