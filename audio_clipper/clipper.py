import pickle
import json

import librosa
import numpy as np
import os
import soundfile as sf
import glob

target_paths = [
    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/world',
    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/hifigan',
    '/mnt/workspace/lijiaqi/phys_vocoder/adver_out/ECAPATDNN_10_0.0004_0.005/*.wav',
    '/mnt/workspace/lijiaqi/phys_vocoder/adver_out/RawNet3_300_0.0004_0.02/*.wav',
    '/mnt/workspace/lijiaqi/phys_vocoder/adver_out/ResNetSE34V2_10_0.0004_0.005/*.wav',
    '/mnt/workspace/lijiaqi/phys_vocoder/adver_out/XVEC_10_0.0004_0.005/*.wav',
    # '/mntnfs/lee_data1/luoyuhao/something/vocoder/waveglow_official',
    # '/mntcephfs/lab_data/wangli/wavernn',
    # '/mntnfs/lee_data1/luoyuhao/something/ASGSR/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10',
    # '/mntnfs/lee_data1/luoyuhao/something/ASGSR/attack/PGD_ResNetSE34V2-ResNetSE34V2_eps-0.005-maxiter-10',
    # '/mntnfs/lee_data1/luoyuhao/something/ASGSR/attack/PGD_RawNet3-RawNet3_eps-0.005-maxiter-10',
    # '/mntnfs/lee_data1/luoyuhao/something/ASGSR/attack/PGD_ECAPATDNN-ECAPATDNN_eps-0.005-maxiter-10',

    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/FAKEBOB_XVEC-20230321145333_eps-0.005-maxiter-1000',
    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/FAKEBOB_ResNetSE34V2-ResNetSE34V2_eps-0.005-maxiter-1000',
    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/FAKEBOB_RawNet3-RawNet3_eps-0.005-maxiter-1000',
    # '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/FAKEBOB_ECAPATDNN-ECAPATDNN_eps-0.005-maxiter-1000',

    # '/mntnfs/lee_data1/wangli/ASGSR/Ensemble/ASVspoof2019/Ensemble_ECAPATDNN-RawNet3-ResNetSE34V2_eps-0.005-maxiter-30',
    # '/mntnfs/lee_data1/wangli/ASGSR/Ensemble/ASVspoof2019/Ensemble_ResNetSE34V2-XVEC-RawNet3_eps-0.005-maxiter-30',
    # '/mntnfs/lee_data1/wangli/ASGSR/Ensemble/ASVspoof2019/Ensemble_ECAPATDNN-XVEC-RawNet3_eps-0.005-maxiter-30',
    # '/mntnfs/lee_data1/wangli/ASGSR/Ensemble/ASVspoof2019/Ensemble_ECAPATDNN-XVEC-ResNetSE34V2_eps-0.005-maxiter-30',
]

# target_path = '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/attack/PGD_XVEC-20230321145333_eps-0.005-maxiter-10'
# target_path = '/mntnfs/lee_data1/wangli/ASGSR/all_asvspoof2019_used_wavs'

def concatenate_audio(audio_list):
    return np.concatenate(audio_list, axis=0)
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

def main():
    for target_path in target_paths:
        generate_audio(target_path, glob.glob(target_path), 1)

main()