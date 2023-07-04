import os.path

from torch.utils.data import Dataset
import torchaudio

class ASVspoof2019(Dataset):
    def __init__(self, data_file, train_path, dev_path, eval_path):

        self._flist = []
        self.train_path = train_path
        self.dev_path = dev_path
        self.eval_path = eval_path

        self.fnamepair_to_idx = dict()  # {enroll_fname}_{eval_fname}

        with open(data_file, 'r') as f:
            for line in f:
                enroll_speaker = line.strip().split(' ')[0]
                enroll_file = line.strip().split(' ')[1]
                eval_speaker = line.strip().split(' ')[2]
                eval_file = line.strip().split(' ')[3]
                enroll_file_path = self.full_path(enroll_file)
                eval_file_path = self.full_path(eval_file)
                self.fnamepair_to_idx[f'{enroll_file}_{eval_file}'] = len(self._flist)
                self._flist.append([enroll_file_path, eval_file_path, enroll_file, eval_file])

    def __len__(self):
        return len(self._flist)

    def __getitem__(self, i):
        enroll_file_path = self._flist[i][0]
        eval_file_path = self._flist[i][1]
        enroll_file = self._flist[i][2]
        eval_file = self._flist[i][3]
        enroll_waveform, _ = torchaudio.load(enroll_file_path)
        eval_waveform, _ = torchaudio.load(eval_file_path)

        return enroll_waveform, eval_waveform, 16000, 0, enroll_file, eval_file
    
    # a generator of {enroll_fname}_{eval_fname} pairs, eval_fname is adversarial
    def get_attack_pairs(self, attack_result_file, attack_file_dir, only_success=False):
        # overwrite attackResult file
        attack_result_file = '/mnt/workspace/lijiaqi/phys_vocoder/adver_out/hifigan0_ECAPATDNN_10_0.0004_0.005/attackResult.txt'
        with open(attack_result_file) as f:
            for line in f:
                line = line.strip().split(' ')
                is_success = line[1]

                enroll_fname = line[0][:12]  # id10270-x6uYqmx31kE-00001
                eval_fname = line[0][13:]  # id10270-x6uYqmx31kE-00003_id10273-8cfyJEV7hP8-00004

                eval_file_path = os.path.join(
                    attack_file_dir,
                    line[0] + '.wav',
                )

                if not os.path.isfile(eval_file_path):
                    print('skipping')
                    continue
                eval_waveform, _ = torchaudio.load(eval_file_path)
                enroll_waveform, _ = torchaudio.load(self._flist[self.fnamepair_to_idx[line[0]]][0])
                
                if len(eval_waveform.size()) == 2:
                    eval_waveform = eval_waveform.unsqueeze(1)
                if len(enroll_waveform.size()) == 2:
                    enroll_waveform = enroll_waveform.unsqueeze(1)

                assert(self._flist[self.fnamepair_to_idx[line[0]]][-1] == eval_fname)

                if only_success:
                    if is_success == 'True':
                        yield enroll_waveform, eval_waveform, enroll_fname, eval_fname
                else:
                    yield enroll_waveform, eval_waveform, enroll_fname, eval_fname

    def full_path(self, file):
        if file.split('_')[1] == 'T':   # train
            file_path = os.path.join(self.train_path, file + '.flac')
        elif file.split('_')[1] == 'D':   # dev
            file_path = os.path.join(self.dev_path, file + '.flac')
        elif file.split('_')[1] == 'E':   # eval
            file_path = os.path.join(self.eval_path, file + '.flac')
        else:
            raise ValueError('Unknown file type: {}'.format(file))
        return file_path
