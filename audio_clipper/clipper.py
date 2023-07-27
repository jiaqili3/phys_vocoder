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

# limit hours
MAX_AUDIO_LEN = int(3600 * 16000 * 23.5)

target_paths = [
]
    
target_paths += glob.glob('/mntcephfs/lab_data/wangli/concat/*.wav')
target_paths += glob.glob('/mntcephfs/lab_data/lijiaqi/adver_out/*_UNetMixedLossNormalizedEndToEnd_*/*.wav')
# target_paths += glob.glob('/mntcephfs/lab_data/lijiaqi/adver_out/*_UNetSpecLossEndToEnd_10_*/')
save_path = '/mntcephfs/lab_data/lijiaqi/audio_full/0722'

# target_path = '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10'
# target_path = '/mntnfs/lee_data1/wangli/ASGSR/all_asvspoof2019_used_wavs'

def concatenate_audio(audio_list):
    # return np.concatenate(audio_list, axis=0)
    return torch.cat(audio_list, dim=1)

def generate_audios(target_paths):
    # print(target_paths)
    all_entries = []
    limit = MAX_AUDIO_LEN

    all_entries = target_paths
    # with open('2500_list.txt') as f:
    #     selected_files = f.readlines()
    # for path in target_paths:
    #     for fname in selected_files:
    #         fname = fname.rstrip('\n')
    #         all_entries.append(f'{path}/{fname}')
    # all_entries += 

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