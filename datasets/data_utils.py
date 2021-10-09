import torch
import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    def __init__(self, indices: list, max_frames: int, data_lengths: list, min_batchsize: int, multiplier=1):
        super().__init__(indices)

        self.indices = indices
        self.data_lengths = data_lengths

        self.bucket_sampler = self.bucket_batch_by_length(max_frames * multiplier, min_batchsize)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            yield bucket

    def __len__(self):
        return len(self.bucket_sampler)

    def bucket_batch_by_length(self, max_frames, min_batchsize, min_frames=9):
        """
        Organize indices into batches such that each batch contains maximally max_frames.

        min_batchsize is to ensure each GPU has at least one instance to run. If not, the batch is simply ignored,
        and wait till next epoch after shuffling.
        """
        batch = []
        batches = []

        frames_in_batch = 0

        for idx in self.indices:
            num_frames = max(self.data_lengths[idx], min_frames)

            if frames_in_batch + num_frames > max_frames:

                if len(batch) < min_batchsize:
                    # discard the batch
                    batch = []
                    frames_in_batch = 0
                    continue

                # batch.extend([batch[-1]] * (min_batchsize - len(batch) % min_batchsize))
                batches.append(batch)

                # reset batch stats
                if num_frames > max_frames:
                    # if GPU memory cannot hold the video
                    batch = []
                    frames_in_batch = 0
                else:
                    batch = [idx]
                    frames_in_batch = num_frames
            else:
                batch.append(idx)
                frames_in_batch += num_frames

        return batches


class BucketDataLoaderWrapper:
    def __init__(self, dataset, max_frames, num_workers, pin_memory, min_batchsize, timeout=0):
        self.dataset = dataset
        self.max_frames = max_frames

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout

        self.min_batchsize = min_batchsize
        self.dataloader = self.build_bucketing_dataloader()

    def build_bucketing_dataloader(self):
        self.dataset.resampling()
        indices = self.dataset.ordered_indices()

        # create mini-batches with given size constraints
        bucket_sampler = BucketBatchSampler(indices,
                                            max_frames=self.max_frames,
                                            data_lengths=self.dataset.video_lengths,
                                            min_batchsize=self.min_batchsize
                                            )

        # return the bucketing dataloader
        # don't do shuffling in the dataloader, this is done in ordered_indices()
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_sampler=bucket_sampler,
                                                 collate_fn=pad_collater,
                                                 shuffle=False,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory,
                                                 timeout=self.timeout
                                                 )

        return dataloader

    def shuffle(self):
        self.dataloader = self.build_bucketing_dataloader()


def pad_collater(samples):
    maxlen_inbatch = max([s[0].shape[1] for s in samples])

    collated_imgs = []
    collated_labels = []

    for s in samples:
        img, label = s

        if img.shape[1] < maxlen_inbatch:
            img = torch.from_numpy(pad(img.transpose(0, 1), min_length=maxlen_inbatch)).transpose(0, 1)

        label = label.unsqueeze(0).repeat(maxlen_inbatch, 1)

        collated_imgs.append(img)
        collated_labels.append(label)

    return torch.stack(collated_imgs), torch.stack(collated_labels)


def pad(imgs, min_length=9):
    """
    padding by repeating either the head or tail frame.
    """
    # i3d minimally accepts 9 frames
    if imgs.shape[0] < min_length:
        num_padding = min_length - imgs.shape[0]

        prob = np.random.random_sample()
        if prob > 0.5:
            pad_img = imgs[0]
            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
            padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            pad_img = imgs[-1]
            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
            padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    return padded_imgs
