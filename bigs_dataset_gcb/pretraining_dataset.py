# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
from .bert_dataset_provider import BertDatasetProviderInterface
from concurrent.futures import ProcessPoolExecutor
from enum import IntEnum

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from google.cloud import storage
from google.resumable_media.common import DataCorruption

logger = logging.getLogger(__name__)

client = storage.Client()


class BatchType(IntEnum):
    RANKING_BATCH = 0
    QP_BATCH = 1
    PRETRAIN_BATCH = 2


def torch_long(x):
    return torch.LongTensor(x)


def map_to_torch(encoding):
    encoding = torch_long(encoding)
    encoding.requires_grad_(False)
    return encoding


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(
    bucket_name,
    input_file,
    max_predictions_per_seq,
    num_workers,
    train_batch_size,
    # worker_init,
    data_sampler,
    no_nsp=True,
):
    train_data = pretraining_dataset(
        bucket_name=bucket_name, input_file=input_file, max_predictions_per_seq=max_predictions_per_seq, no_nsp=no_nsp
    )
    train_dataloader = DataLoader(
        train_data,
        sampler=data_sampler(train_data),
        batch_size=train_batch_size,
        num_workers=num_workers,
        # worker_init_fn=worker_init,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader, len(train_data)


class pretraining_dataset(Dataset):
    def __init__(self, bucket_name, input_file, max_predictions_per_seq, no_nsp=True):
        self.bucket_name = bucket_name
        self.input_file = input_file
        self.max_predictions_per_seq = max_predictions_per_seq

        # first download file google bucket
        bucket = client.get_bucket(bucket_name)

        folder_name, file_name = input_file.rsplit('/', 1)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            # start download file
        else:
            old_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith(".hdf5")]
            # check maximal file to avoid too many files
            if len(old_files) > 5:
                for old_file in old_files:
                    os.remove(old_file)

        notDownloadSuccss = True
        while notDownloadSuccss:
            try:
                blob = bucket.blob(input_file)
                blob.reload()
                blob.download_to_filename(input_file)
                notDownloadSuccss = False
            except DataCorruption:
                notDownloadSuccss = True
                # retry download

        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.no_nsp = no_nsp
        if no_nsp:
            keys.remove("next_sentence_labels")
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids] = [
            torch.from_numpy(input[index].astype(np.int64))
            for _, input in enumerate(self.inputs[:5])
        ]
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_predictions_per_seq
        # store number of  masked tokens in index
        padded_mask_indices = torch.nonzero((masked_lm_positions == 0), as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        if self.no_nsp:
            return [
                input_ids,
                input_mask,
                segment_ids,
                masked_lm_labels,
            ]
        else:
            next_sentence_labels = torch.from_numpy(
                np.asarray(self.inputs[-1][index].astype(np.int64))
            )
            return [
                input_ids,
                input_mask,
                segment_ids,
                next_sentence_labels,
                masked_lm_labels,
            ]


class ValidationDataset:
    def __init__(self, args):
        self.global_rank = 0
        self.world_size = 1

        # Initialize dataset files
        dataset_path = args.dataset_path

        # get dataset files in google cloud
        self.bucket_name, file_path = dataset_path.split('/', 1)
        self.dataset_files = [blob.name for blob in client.list_blobs(self.bucket_name, prefix=file_path) if
                              "test" in blob.name]
        self.dataset_files.sort()
        self.num_files = len(self.dataset_files)
        if self.global_rank == 0:
            logger.info(f"ValidationDataset - Initialization:  num_files = {self.num_files}")
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.no_nsp = True

    def get_validation_set(self, index):
        file_index = index % self.num_files
        input_file = self.dataset_files[file_index]
        validation_data = pretraining_dataset(
            bucket_name=self.bucket_name,
            input_file=input_file,
            max_predictions_per_seq=self.max_predictions_per_seq,
            no_nsp=True,
        )
        logger.info(f"ValidationDataset - shard {file_index} - length {len(validation_data)}")
        return validation_data


class PreTrainingDataset(BertDatasetProviderInterface):
    def __init__(self, args, data_prefix="train"):
        self.num_workers = args.num_workers
        self.max_predictions_per_seq = args.max_predictions_per_seq
        assert data_prefix in ["train", "test"], "data_prefix must be [train|test]"

        self.train_batch_size = args.train_batch_size

        self.global_rank = 0
        self.world_size = 1

        # Initialize dataset files
        dataset_path = args.dataset_path

        # get dataset files in google cloud
        self.bucket_name, file_path = dataset_path.split('/', 1)
        self.dataset_files = [blob.name for blob in client.list_blobs(self.bucket_name, prefix=file_path) if data_prefix in blob.name]

        assert len(self.dataset_files) > 0, "No train/test*.hdf5 files found in given dataset path"
        if data_prefix == "train":
            self.dataset_files.sort()
        random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)
        self.data_sampler = RandomSampler

        self.dataset_future = None
        self.pool = ProcessPoolExecutor(1)

        logger.info(f"PreTrainingDataset - Initialization:  num_files = {self.num_files}")
        self.no_nsp = True

    def get_shard(self, index):
        if self.dataset_future is None:
            data_file = self._get_shard_file(index)
            self.train_dataloader, sample_count = create_pretraining_dataset(
                bucket_name=self.bucket_name,
                input_file=data_file,
                max_predictions_per_seq=self.max_predictions_per_seq,
                num_workers=self.num_workers,
                train_batch_size=self.train_batch_size,
                # worker_init=self.worker_init,
                data_sampler=self.data_sampler,
                no_nsp=self.no_nsp,
            )
        else:
            self.train_dataloader, sample_count = self.dataset_future.result(timeout=None)

        return self.train_dataloader, sample_count

    def release_shard(self, index):
        del self.train_dataloader

    def prefetch_shard(self, index):
        data_file = self._get_shard_file(index)
        self.dataset_future = self.pool.submit(
            create_pretraining_dataset,
            self.bucket_name,
            data_file,
            self.max_predictions_per_seq,
            self.num_workers,
            self.train_batch_size,
            # self.worker_init,
            self.data_sampler,
            self.no_nsp,
        )

    def get_batch(self, batch_iter):
        return batch_iter

    def prefetch_batch(self):
        pass

    def _get_shard_file(self, shard_index):
        file_index = self._get_shard_file_index(shard_index, self.global_rank)
        return self.dataset_files[file_index % self.num_files]

    def _get_shard_file_index(self, shard_index, global_rank):
        if dist.is_initialized() and self.world_size > self.num_files:
            remainder = self.world_size % self.num_files
            file_index = (shard_index * self.world_size) + global_rank + (remainder * shard_index)
        else:
            file_index = shard_index * self.world_size + global_rank

        return file_index % self.num_files
