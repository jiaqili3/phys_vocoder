import pickle
import json
import numpy as np
import torchaudio
import os

# device = 'iphone'
devices = [
    'iphone',
    'mate50',
]
from_device = 'k40s'
distance = '0.5m'

vocoders = [
    'diffwave',
    'hifigan',
    'waveglow',
    'wavernn',
    'world',
]

parts = ['1', '2']

# for device in devices:
#     for vocoder in vocoders:
#         for part in parts:

device = 'mate50'
vocoder = 'diffwave'
part = '2'

save_folder = f'D:/device_recordings/recovered/{device}_{vocoder}_{from_device}_{distance}/'
# save_folder = f'D:/device_recordings/recovered/test_split/'

os.makedirs(save_folder, exist_ok=True)

# full_audio_path = r"D:\device_recordings\vocoders\hifigan\2\pulse_hifigan_audio_full_2.wav"
full_audio_path = f"D:/device_recordings/vocoders/{vocoder}/{part}/{device}_{vocoder}_{from_device}_{distance}_{part}.wav"
pkl_path = f'./audio_meta_{vocoder}_{part}.pkl'

# time after the pulse is 0.5s
extension_frames = int(0.5 * 16000)


def main():
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)




    # load the recorded audio
    # the recording should be edited out the start blank space
    rec_full_waveform, sr = torchaudio.load(full_audio_path)



    curr_pulse_point_start = 0
    # find the largest pulse in the first 1s and reassign the start point
    largest = 0
    largest_idx = 0
    for i in range(int(sr * 1)):
        # find the sample point with the largest amplitude
        if rec_full_waveform[0, i] > largest:
            largest = rec_full_waveform[0, i]
            largest_idx = i
    curr_pulse_point_start = largest_idx
    print(f'start pulse point: {curr_pulse_point_start}')

    for entry in metadata:
        audio_path = entry['audio_path']
        audio_fname = audio_path.split('/')[-1]


        time_duration = entry['time_duration']
        frame_duration = entry['sample_points']
        if (int(time_duration) == 20):
            frame_duration = 20 * 16000
        
        assert(sr == entry['sample_rate'])
        assert(sr == 16000)

        if audio_path != 'blank' and audio_path != 'pulse':

            waveform_slice = rec_full_waveform[:, curr_pulse_point_start+extension_frames:curr_pulse_point_start+frame_duration+extension_frames]
            # print(curr_sample_point, curr_sample_point+frame_duration+extension_frames)
            # print(waveform_slice.shape)
            torchaudio.save(save_folder + audio_fname, waveform_slice.reshape(1,-1), sr)
            print(f'write {audio_fname} to {save_folder + audio_fname}, duration: {time_duration}')

            # get the next pulse point in the 0.8s surrounding of the estimated pulse point
            estimated_pulse_point_start = curr_pulse_point_start+frame_duration+extension_frames+extension_frames
            # find the sample point with the largest amplitude
            largest = 0
            largest_idx = 0
            for i in range(estimated_pulse_point_start-int(sr*0.4), estimated_pulse_point_start+int(sr*0.4)):
                if rec_full_waveform[0, i] > largest:
                    largest = rec_full_waveform[0, i]
                    largest_idx = i
            curr_pulse_point_start = largest_idx
            print(f'diff: {curr_pulse_point_start - estimated_pulse_point_start}')
            assert(curr_pulse_point_start - estimated_pulse_point_start < 1000)
            assert(largest_idx != 0)


main()