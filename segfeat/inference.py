from .model import Segmentor
import torch
import numpy as np
import librosa
from scipy.stats import mode


def mfcc_dist(mfcc):
    """mfcc_dist
    calc 4-dimensional dist features like in HTK

    :param mfcc:
    """
    d = []
    for i in range(2, 9, 2):
        pad = int(i/2)
        d_i = np.concatenate([np.zeros(pad), ((mfcc[:, i:] - mfcc[:, :-i]) ** 2).sum(0) ** 0.5, np.zeros(pad)], axis=0)
        d.append(d_i)
    return np.stack(d)

def extract_features(wav, sr, feats='mfcc', n_fft=160, hop_length=160, n_mels=40, 
                    n_mfcc=13, normalize=False, delta_feats=True, dist_feats=True):
    '''
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length,
        n_mels=hparams.rnn_input_size


    '''
    # print(wav.shape, sr)

    # extract mel-spectrogram
    if feats == 'mel':
        spect = librosa.feature.melspectrogram(wav,
                                               sr=sr,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               n_mels=n_mels)
    # extract mfcc
    elif feats == 'mfcc':
        spect = librosa.feature.mfcc(wav,
                                     sr=sr,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     n_mels=n_mels,
                                     n_mfcc=n_mfcc)
        if normalize:
            spect = (spect - spect.mean(0)) / spect.std(0)
        if delta_feats:
            delta  = librosa.feature.delta(spect, order=1)
            delta2 = librosa.feature.delta(spect, order=2)
            spect  = np.concatenate([spect, delta, delta2], axis=0)
        if dist_feats:
            dist = mfcc_dist(spect)
            spect  = np.concatenate([spect, dist], axis=0)
    else:
        raise Exception("no features specified!")

    spect = torch.transpose(torch.FloatTensor(spect), 0, 1)
    return spect

class SavedSegmentor:
    def __init__(self, config, device, chkpt_path):
        '''
            Wrapped around saved model checkpoint

            Parameters:
                config: dict with configuration parameters (same as used in main.py for training). See configs/model_params.json for example
                device: device for inference (cuda, cpu, cuda:0, etc)
                chkpt_path: path to pytorch checkpoint
            
        '''

        config['rnn_input_size'] = {'mfcc':  config['n_mfcc'] * (3 if config['delta_feats'] else 1) + (4 if config['dist_feats'] else 0),
                                        'mel':   config['n_mels'],
                                        'spect': config['n_fft'] / 2 + 1}[config['feats']]
        
        #currently supports only timit
        config['n_classes'] = {'timit': 39,
                            'buckeye': 40}[config['dataset']]

        if config['dataset'] == 'timit':
            self.words2idx = {'ae': 29, 'ah': 25, 'ao': 8, 'aw': 1, 'ay': 12, 'b': 9,
                        'ch': 38, 'd': 6, 'dh': 19, 'dx': 32, 'eh': 0, 'el': 11,
                        'en': 31, 'er': 16, 'ey': 5, 'f': 18, 'g': 34, 'hh': 28,
                        'ih': 26, 'iy': 20, 'jh': 23, 'k': 7, 'm': 3, 'ng': 27,
                        'ow': 24, 'oy': 2, 'p': 17, 'r': 4, 's': 36, 'sh': 33,
                        'sil': 10, 't': 35, 'th': 14, 'uh': 37, 'uw': 30,
                        'v': 15, 'w': 13, 'y': 21, 'z': 22}

            self.idx2word = {val: key for key, val in self.words2idx.items()}
        else:
            raise NotImplementedError('Only models trained on timit dataset are supported')

        segmentor = Segmentor(config)

        model_dict = segmentor.state_dict()
        weights = torch.load(chkpt_path, map_location='cpu')["state_dict"]
        weights = {k.replace("segmentor.", ""): v for k,v in weights.items()}
        weights = {k: v for k,v in weights.items() if k in model_dict and model_dict[k].shape == weights[k].shape}
        model_dict.update(weights)
        segmentor.load_state_dict(model_dict)
        segmentor.to(device)

        assert len(weights) == len(model_dict)

        self.segmentor = segmentor
        self.device = device

    def __call__(self, wav, sample_rate):
        '''
            Segment wav data using the model

            Parameters:
                wav: numpy array
                sample_rate: self-explanatory

            You can obtain these parameters with `wav, sr = sound_file.read(filename)`
            
            Returns:
                List[Tuple(left border index, right border index, phoneme)]

        '''
        
        feats = extract_features(wav, sample_rate)

        with torch.no_grad():
            self.segmentor.eval()

            length = torch.LongTensor([feats.shape[0]])
            result = self.segmentor(feats.unsqueeze(0).to(self.device), length)

            phon = result['cls_out'].cpu().numpy().argmax(2)[0]
            bins = result['pred'][0]

            chunks = []
            
            for left, right in zip(bins[:-1], bins[1:]):
                # stat = np.unique(phon[left: right], return_counts=True)
                index = mode(phon[left: right]).mode[0]
                chunks.append((left, right, self.idx2word[index]))

            return chunks