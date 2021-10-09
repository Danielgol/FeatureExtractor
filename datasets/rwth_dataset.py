import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl

from datasets import data_utils


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def cv2_load_frames_from_video(video_path, start, end):
    vidcap = cv2.VideoCapture(video_path)

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(end - start):
        success, img = vidcap.read()

        w, h, c = img.shape

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        # normalization
        img = (img / 255.) * 2 - 1

        # opencv by default loads frames / videos as BGR, here we convert to RGB.
        frames.append(img[:, :, [2, 1, 0]])

    return np.asarray(frames, dtype=np.float32)


def build_vocab(entries):
    d = dict()

    for e in entries:
        g = e['gloss']

        if g not in d:
            d[g] = len(d)

    return d


def make_dataset(split_file, vocab_file):
    dataset = []

    with open(split_file, 'r') as f:
        data = json.load(f)

    with open(vocab_file, 'r') as f:
        lines = f.read().split('\n')

    data = [d for d in data if d['gloss'] not in ['__ON__', '__OFF__']]

    for item in data:
        dataset.append({'label': item['label'],
                        'start': item['start'],
                        'end': item['end'],
                        'iden': item['video'],
                        'src': item['src']
                        })

    return dataset, len(lines)


class BalancedDataset(data_utl.Dataset):

    def __init__(self, split_file, vocab_file, root_cslr, root_slt, is_balance, transforms=None, is_train=False):
        self.data, self.num_class = make_dataset(split_file, vocab_file)
        self.video_lengths = np.array([d['end'] - d['start'] for d in self.data])

        self.is_balance = is_balance

        self.transforms = transforms
        self.root_cslr = root_cslr
        self.root_slt = root_slt

        self.is_train = is_train

        self.inst_freq, self.label_freq = self.build_label2freq()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data[index]
        label, start, end, iden = item['label'], item['start'], item['end'], item['iden']

        if item['src'] == 'phoenix-2014':
            video_path = os.path.join(self.root_cslr, iden + '.mp4')
        elif item['src'] == 'phoenix-2014-T':
            video_path = os.path.join(self.root_slt, iden + '.mp4')
        else:
            raise ValueError('invalid src {}'.format(item['src']))

        imgs = cv2_load_frames_from_video(video_path, start, end)

        if self.transforms:
            imgs = self.transforms(imgs)

        imgs = data_utils.pad(imgs)

        ret_img = video_to_tensor(imgs)
        ret_lab = self.one_hot_label(label)

        return ret_img, ret_lab

    def __len__(self):
        return len(self.data)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.is_train:
            indices = np.random.permutation(self.resample_indices)
        else:
            indices = np.array(self.resample_indices)

        return indices[np.argsort(self.video_lengths[indices], kind='mergesort')]

    def resampling(self, oversampling_mul=2, undersampling_mul=0.7):
        if self.is_train and self.is_balance:
            # max_freq = np.max(self.inst_freq)
            #
            # # calculate imbalance ratio
            # imbalance_ratio = max_freq / self.inst_freq
            # avg_ratio = np.average(imbalance_ratio)
            #
            # # draw extra samples from an oversampling distribution
            # oversampling_ratio = imbalance_ratio.copy()
            # oversampling_ratio[oversampling_ratio < avg_ratio] = 0
            # # normalization
            # oversampling_dist = lin_normalize(oversampling_ratio)
            # oversampling_indices = np.random.choice(len(oversampling_dist), oversampling_mul * len(self.data), p=oversampling_dist)
            #
            # # also drop highly-frequent samples from distribution
            # drop_ratio = imbalance_ratio.copy()
            # drop_ratio = 1 / drop_ratio
            #
            # drop_ratio[drop_ratio < 1 / avg_ratio] = 0
            # # normalization
            # drop_dist = lin_normalize(drop_ratio)
            # drop_indices = np.random.choice(len(drop_dist), int(undersampling_mul * len(self.data)), p=drop_dist)
            #
            # resample_indices = []
            #
            # # under-sampling
            # for idx, d in enumerate(self.data):
            #     if idx not in drop_indices:
            #         resample_indices.append(idx)
            #
            # # over-sampling
            # for idx in oversampling_indices:
            #     resample_indices.append(idx)

            count = dict()
            resample_indices = []

            for d in self.data:
                count[d['label']] = 0

            for idx, d in enumerate(self.data):
                label = d['label']

                if count[label] < 100:
                    resample_indices.append(idx)

                    count[label] += 1
        else:
            resample_indices = list(range(len(self.data)))

        self.resample_indices = resample_indices

        # temp_freq = np.zeros(self.num_class)
        # for entry in resample_indices:
        #         temp_freq[self.data[entry]['label']] += 1
        #
        # import matplotlib.pyplot as plt; plt.rcdefaults()
        #
        # sorted_indices = sorted(np.arange(self.num_class), key=lambda x: temp_freq[x])
        # plt.bar(np.arange(len(temp_freq)), temp_freq[sorted_indices], align='center', alpha=0.5)
        #
        # sorted_indices_2 = sorted(np.arange(self.num_class), key=lambda x: self.label_freq[x])
        # plt.bar(np.arange(len(self.label_freq)), self.label_freq[sorted_indices_2], align='center', alpha=0.5)
        # plt.show()
        # import pdb; pdb.set_trace()

    def one_hot_label(self, label_index):
        label = np.zeros(shape=self.num_class)
        label[label_index] = 1.0

        return torch.tensor(label)

    def build_label2freq(self):
        label_freq = np.zeros(self.num_class)
        inst_freq = np.zeros(len(self.data))

        for entry in self.data:
            label_freq[entry['label']] += 1

        for i, entry in enumerate(self.data):
            inst_freq[i] += label_freq[entry['label']]

        return inst_freq, label_freq


def lin_normalize(x):
    return x / sum(x)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    imgs = np.ones(shape=(9, 256, 256, 3))
    label = np.ones(shape=(9, 1, 100))

    BalancedDataset.pad_wrap(imgs, label, total_frames=25)
