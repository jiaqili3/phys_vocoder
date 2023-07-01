import pickle
import numpy as np
import torchaudio
import os
import torch

models = [
    'xvec',
    # 'rawnet3',
    # 'ecapatdnn',
    # 'resnet34v2',
]

devices = [
    'mate50',
    'iphone',
]

from_devices = [
    'k40s',
    # 'hwsound',
]
model_to_meta = {
    'xvec': 'PGD_XVEC-20230321145333_eps-0.005-maxiter-10',
    'resnet34v2': 'PGD_ResNetSE34V2-ResNetSE34V2_eps-0.005-maxiter-10',
    'rawnet3': 'PGD_RawNet3-RawNet3_eps-0.005-maxiter-10',
    'ecapatdnn': 'PGD_ECAPATDNN-ECAPATDNN_eps-0.005-maxiter-10',
}

for model in models:
    for device in devices:
        for from_device in from_devices:


            # device = 'mate50'
            # from_device = 'k40s'
            # model = 'resnet34v2'
            distance = '0.5m'
            attack_type = 'pgd' # or ensemble


            save_folder = f'D:/device_recordings/recovered_pulse/{device}_{attack_type}_{model}_{from_device}_{distance}/'

            os.makedirs(save_folder, exist_ok=True)

            full_audio_path = f'D:/device_recordings/pulse_{attack_type}_{model}/pulse_{device}_{attack_type}_{model}_{from_device}_{distance}.wav'
            assert os.path.isfile(full_audio_path)

            # time after the pulse is 0.5s
            extension_frames = int(0.5 * 16000)


            def main():
                try:
                    metadata_path = f'./audio_meta_{model_to_meta[model]}.pkl'
                    assert os.path.isfile(metadata_path)
                except:
                    metadata_path = f'audio_clipper/audio_meta_{model_to_meta[model]}.pkl'
                    assert os.path.isfile(metadata_path)
                with open(metadata_path, 'rb') as f:
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

