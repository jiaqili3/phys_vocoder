import pickle
import json
import numpy as np
import torchaudio
import torchaudio.functional as F
import os
from pathlib import Path

# put this to the second at the beginning of the first pulse
# make it 0 if the audio's already starting with the first pulse
offset = 44 * 16000

# which folder to save the recovered audios
save_folder = f'/mntcephfs/lab_data/lijiaqi/phys_vocoder_recordings/iphone_RawNet3_UNetEndToEnd_300_0.0004_0.02'
os.makedirs(save_folder, exist_ok=True)

# the path to the full audio, must be ending with ".wav"
full_audio_path = f"/home/lijiaqi/phys_vocoder/audio_clipper/iphone.wav"

# the path to the pkl file
pkl_path = f'/home/lijiaqi/phys_vocoder/audio_clipper/audio_meta_RawNet3_UNetEndToEnd_300_0.0004_0.02_1.pkl'

# config is done

# time after the pulse is 0.5s
extension_frames = int(0.5 * 16000)
def main():
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)




    # load the recorded audio
    # the recording should be edited out the start blank space
    rec_full_waveform, sr = torchaudio.load(full_audio_path)
    rec_full_waveform = F.resample(rec_full_waveform, sr, 16000)
    rec_full_waveform = rec_full_waveform[:, offset:]

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
            save_path = os.path.join(save_folder, audio_fname)
            torchaudio.save(save_path, waveform_slice.reshape(1,-1), sr)
            print(f'write {audio_fname} to {save_path}, duration: {time_duration}')

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