import os
audio_paths = [
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac',
]
audio_outpaths = [
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/wav',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/wav',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/wav',
]

for path in audio_outpaths:
    os.makedirs(path, exist_ok=True)

for i, audio_path in enumerate(audio_paths):
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.flac'):
                audio_outpath = os.path.join(audio_outpaths[i], file[:-5] + '.wav')
                if not os.path.exists(audio_outpath):
                    os.system('ffmpeg -i %s %s' % (os.path.join(root, file), audio_outpath))
                    print(audio_outpath)