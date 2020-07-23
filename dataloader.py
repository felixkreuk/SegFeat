import os
import pickle
import random
from multiprocessing import Pool
from os.path import basename, join

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from boltons.fileutils import iter_find_files
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length

    :param batch:
    """
    # get sequence lengths
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    lengths = [t[3] for t in batch]

    # pad and stack
    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segs, labels, lengths


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


def phoneme_lebels_to_frame_labels(segmentation, phonemes):
    """
    replicates phonemes to frame-wise labels
    example:
        segmentation - [0, 3, 4]
        phonemes - [a, b]
        returns - [a, a, a, b]

    :param segmentation:
    :param phonemes:
    """
    segmentation = torch.LongTensor(segmentation)
    return torch.cat([torch.LongTensor([l]).repeat(t) for (l, t) in zip(phonemes, segmentation[1:] - segmentation[:-1])])


def segmentation_to_binary_mask(segmentation):
    """
    replicates boundaries to frame-wise labels
    example:
        segmentation - [0, 3, 5]
        returns - [1, 0, 0, 1, 0, 1]

    :param segmentation:
    :param phonemes:
    """
    mask = torch.zeros(segmentation[-1] + 1).long()
    for boundary in segmentation[1:-1]:
        mask[boundary] = 1
    return mask


def extract_features(wav_file, hparams):
    wav, sr = sf.read(wav_file)

    # extract mel-spectrogram
    if hparams.feats == 'mel':
        spect = librosa.feature.melspectrogram(wav,
                                               sr=sr,
                                               n_fft=hparams.n_fft,
                                               hop_length=hparams.hop_length,
                                               n_mels=hparams.rnn_input_size)
    # extract mfcc
    elif hparams.feats == 'mfcc':
        spect = librosa.feature.mfcc(wav,
                                     sr=sr,
                                     n_fft=hparams.n_fft,
                                     hop_length=hparams.hop_length,
                                     n_mels=hparams.n_mels,
                                     n_mfcc=hparams.n_mfcc)
        if hparams.normalize:
            spect = (spect - spect.mean(0)) / spect.std(0)
        if hparams.delta_feats:
            delta  = librosa.feature.delta(spect, order=1)
            delta2 = librosa.feature.delta(spect, order=2)
            spect  = np.concatenate([spect, delta, delta2], axis=0)
        if hparams.dist_feats:
            dist = mfcc_dist(spect)
            spect  = np.concatenate([spect, dist], axis=0)
    else:
        raise Exception("no features specified!")

    spect = torch.transpose(torch.FloatTensor(spect), 0, 1)
    return spect


def random_trim(spect, seg):
    start_trim = random.randint(1, seg[1].item() - 1)
    end_trim = random.randint(1, (seg[-1] - seg[-2]).item() - 1)
    spect = spect[start_trim: -end_trim]
    seg[1:] -= start_trim
    seg[-1] = seg[-1] - end_trim

    return spect, seg


def get_onset_offset(segmentations):
    search_start, search_end = float("inf"), 0
    for seg in segmentations:
        start, end = seg[0], seg[-1]
        if start < search_start:
            search_start = start
        if end > search_end:
            search_end = end
    return search_start, search_end


def is_valid_lens(spect, seg, hparams):
    sizes = seg[1:] - seg[:-1]
    if len(spect) <= hparams.max_len and \
       sizes.min().item() >= hparams.min_seg_size and \
       sizes.max().item() <= hparams.max_seg_size:
       return True
    return False


class TimitDictionary():
    def __init__(self):
        dict_file = "dicts/phoneme_39_dict.txt"
        with open(dict_file, 'r') as f:
            self.phn_reduction_dict = {line.split(' ')[0].strip(): line.split(' ')[1].strip() for line in f.readlines()}
        self.words = set(self.phn_reduction_dict.values())
        self.n_words = len(self.words)
        self._words2idx = {'ae': 29, 'ah': 25, 'ao': 8, 'aw': 1, 'ay': 12, 'b': 9,
                           'ch': 38, 'd': 6, 'dh': 19, 'dx': 32, 'eh': 0, 'el': 11,
                           'en': 31, 'er': 16, 'ey': 5, 'f': 18, 'g': 34, 'hh': 28,
                           'ih': 26, 'iy': 20, 'jh': 23, 'k': 7, 'm': 3, 'ng': 27,
                           'ow': 24, 'oy': 2, 'p': 17, 'r': 4, 's': 36, 'sh': 33,
                           'sil': 10, 't': 35, 'th': 14, 'uh': 37, 'uw': 30,
                           'v': 15, 'w': 13, 'y': 21, 'z': 22}
        self._idx2words = {v:k for k,v in self._words2idx.items()}

    def word2idx(self, word):
        word = self.phn_reduction_dict[word]
        return self._words2idx[word]

    def words2idx(self, words):
        return [self.word2idx(word) for word in words]


class BuckeyeDictionary():
    def __init__(self):
        dict_file = "dicts/phoneme_39_dict.txt"
        with open(dict_file, 'r') as f:
            self.phn_reduction_dict = {line.split(' ')[0].strip(): line.split(' ')[1].strip() for line in f.readlines()}
        self.n_words = 0
        self._words2idx = {}

    def word2idx(self, word):
        if word in self.phn_reduction_dict:
            word = self.phn_reduction_dict[word]
        elif word in ['SIL', 'VOCNOISE', 'NOISE']:
            word = 'sil'
        else:
            word = 'vn'

        if word in self._words2idx:
            return self._words2idx[word]
        else:
            self._words2idx[word] = self.n_words
            self.n_words += 1
            return self._words2idx[word]

    def words2idx(self, words):
        return [self.word2idx(word) for word in words]


class WavPhnDataset(Dataset):
    def __init__(self, path, hparams):
        self.hparams = hparams
        self.wav_path = path
        super(WavPhnDataset, self).__init__()

    @staticmethod
    def get_datasets(hparams):
        raise NotImplementedError

    def process_file(self, wav_path):
        phn_path = wav_path.replace("wav", "phn")

        # load audio
        spect = extract_features(wav_path, self.hparams)

        # load labels -- segmentation and phonemes
        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            # get segment times
            times   = torch.FloatTensor([0] + list(map(lambda line: int(line[1]), lines)))
            wav_len = times[-1]
            times   = (times / wav_len * (len(spect) - 1)).long()

            # get phonemes in each segment (for K times there should be K+1 phonemes)
            phonemes = list(map(lambda line: line[2].strip(), lines))
            phonemes = self.vocab.words2idx(phonemes)

        # check if audio len and segment sizes are ok
        if is_valid_lens(spect, times, self.hparams):
            return spect, times.tolist(), phonemes

        return None

    def _make_dataset(self):
        files = []
        wavs = list(iter_find_files(self.wav_path, "*.wav"))
        if self.hparams.devrun:
            wavs = wavs[:self.hparams.devrun_size]

        for wav in tqdm(wavs, desc="loading data into memory"):
            res = self.process_file(wav)
            if res is not None:
                files.append(res)

        return files

    def __getitem__(self, idx):
        spect, seg, phonemes = self.data[idx]

        if self.hparams.random_trim:
            spect, seg = random_trim(spect, seg)

        return spect, seg, phonemes, spect.shape[0]

    def __len__(self):
        return len(self.data)


class TimitDataset(WavPhnDataset):
    def __init__(self, path, hparams):
        super(TimitDataset, self).__init__(path, hparams)
        self.vocab = TimitDictionary()
        self.data = self._make_dataset()

    @staticmethod
    def get_datasets(hparams):
        train_dataset = TimitDataset(join(hparams.wav_path, 'train'),
                                     hparams)
        test_dataset  = TimitDataset(join(hparams.wav_path, 'test'),
                                     hparams)

        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - hparams.val_ratio))
        val_split   = train_len - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split, val_split])
        logger.info(f"split timit from {train_len} to train {train_split}, valid {val_split}")

        return train_dataset, val_dataset, test_dataset


class BuckeyeDataset(WavPhnDataset):
    def __init__(self, path, hparams):
        super(BuckeyeDataset, self).__init__(path, hparams)
        self.vocab = BuckeyeDictionary()
        self.data = self._make_dataset()

    @staticmethod
    def get_datasets(hparams):
        train_dataset = BuckeyeDataset(join(hparams.wav_path, 'train'),
                                       hparams)
        val_dataset   = BuckeyeDataset(join(hparams.wav_path, 'val'),
                                       hparams)
        test_dataset  = BuckeyeDataset(join(hparams.wav_path, 'test'),
                                       hparams)

        return train_dataset, val_dataset, test_dataset
