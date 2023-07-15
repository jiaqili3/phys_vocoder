import pickle
import json

import librosa
import numpy as np
import os
import soundfile as sf
import glob
import pdb
from pathlib import Path
import torchaudio
import torch

# 19 hours
MAX_AUDIO_LEN = int(3600 * 16000 * 19.6)

target_paths = [
]
    
target_paths += glob.glob('/mntcephfs/lab_data/lijiaqi/adver_out/*_None_10_*/')
target_paths += glob.glob('/mntcephfs/lab_data/lijiaqi/adver_out/*_UNetSpecLossEndToEnd_10_*/')
save_path = '/mntcephfs/lab_data/lijiaqi/audio_full/0715'

# target_path = '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10'
# target_path = '/mntnfs/lee_data1/wangli/ASGSR/all_asvspoof2019_used_wavs'

def concatenate_audio(audio_list):
    # return np.concatenate(audio_list, axis=0)
    return torch.cat(audio_list, dim=1)
def generate_audio(target_path, entries, idx=1):
    
    path_ending = target_path.split('/')[-2]

    metadata = []
    audio_list = []
    sr = 16000
    # create a waveform with 200Hz frequency, at 16,000Hz sampling rate
    # for 20 second
    pulse = np.cos(2 * np.pi * 200 * np.arange(16000*20) / 16000)
    pulse.dtype = np.float64

    blank_after_duration = 1.0
    blank_space_frames = int(sr * blank_after_duration)

    blank_space_waveform = np.zeros((blank_space_frames,), dtype=np.float64)
    blank_space_waveform[blank_space_frames // 2] = 1.0
    blank_space_waveform[blank_space_frames // 2+1] = 1.0

    for i in range(int(20 / blank_after_duration)):
        audio_list.append(blank_space_waveform)
    audio_list.append(pulse)
    audio_list.append(blank_space_waveform)
    metadata.append({
        'audio_path': 'pulse',
        'time_duration': 20,
        'sample_points': 16000*20,
        'sample_rate': 16000,
    })
    metadata.append({
        'audio_path': 'blank',
        'time_duration': blank_space_frames / sr,
        'sample_points': blank_space_frames,
        'sample_rate': sr,
    })
    # for dir in os.listdir(target_path):
    #     print(dir)
    #     dir = os.path.abspath(f'{target_path}/{dir}')
    #     if os.path.isdir(dir):
    for entry in entries:
        # entry = os.path.abspath(f'{target_path}/{entry}')
        if entry.endswith('.wav') or entry.endswith('.flac'):
            print(entry)

            waveform, sr = sf.read(entry)
            audio_list.append(waveform)

            audio_list.append(blank_space_waveform)

            metadata.append({
                'audio_path': entry,
                'time_duration': waveform.shape[0] / sr,
                'sample_points': waveform.shape[0],
                'sample_rate': sr,
            })
            metadata.append({
                'audio_path': 'blank',
                'time_duration': blank_space_frames / sr,
                'sample_points': blank_space_frames,
                'sample_rate': sr,
            })
    audio = concatenate_audio(audio_list)
    os.makedirs('audio_full', exist_ok=True)
    sf.write(f'./audio_full/pulse_{path_ending}_audio_full_{idx}.wav', audio, sr)
    # output metadata
    print(len(metadata))
    with open(f'audio_meta_{path_ending}_{idx}.pkl', 'wb') as f:
        pickle.dump(metadata, f)



def generate_audios(target_paths):
    print(target_paths)
    all_entries = []
    limit = MAX_AUDIO_LEN

    with open('2500_list.txt') as f:
        selected_files = f.readlines()
    for path in target_paths:
        for fname in selected_files:
            fname = fname.rstrip('\n')
            all_entries.append(f'{path}/{fname}')

    idx = 0
    while all_entries != []:
        idx += 1
        audio_len = 0
        audio_list = []
        metadata = []
        blank_space_frames = int(16000*0.5)
        blank_space_waveform = torch.zeros((1,blank_space_frames), dtype=torch.float32)
        blank_space_waveform[:,blank_space_frames // 2] = 1.0

        for i in range(40):
            audio_list.append(blank_space_waveform)
        while audio_len < limit and all_entries != []:
            audio_path = all_entries.pop()
            audio_path = Path(audio_path)
            entry_token = f'{audio_path.parent.name}/{audio_path.name}'
            print(entry_token)
            # pdb.set_trace()
            waveform, sr = torchaudio.load(str(audio_path))
            assert sr == 16000
            audio_list.append(waveform)
            audio_list.append(blank_space_waveform)
            audio_len += waveform.shape[1] + blank_space_waveform.shape[1]

            metadata.append({
                'audio_path': entry_token,
                'time_duration': waveform.shape[1] / sr,
                'sample_points': waveform.shape[1],
                'sample_rate': sr,
            })
            metadata.append({
                'audio_path': 'blank',
                'time_duration': blank_space_frames / sr,
                'sample_points': blank_space_frames,
                'sample_rate': sr,
            })
        # save
        audio = concatenate_audio(audio_list)
        os.makedirs(save_path, exist_ok=True)
        torchaudio.save(f'{save_path}/{Path(save_path).name}_{idx}.wav', audio, 16000, bits_per_sample=16, encoding='PCM_S')
        print('saved to', f'{save_path}/{Path(save_path).name}_{idx}.wav')
        print('audio len:', audio_len/16000/3600)
        with open(f'audio_meta_{Path(save_path).name}_{idx}.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print('metadata dumped to', f'audio_meta_{Path(save_path).name}_{idx}.pkl')


def main():
    generate_audios(target_paths)

main()