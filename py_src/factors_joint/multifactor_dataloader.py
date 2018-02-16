# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2017-10-23 20:47:21
# @Last Modified by:   Jiaqi guo
# @Last Modified time: 2018-01-23 14:37:21

import numpy as np
import torch
from ..utils import kaldi_io
import itertools
from torch.utils.data import TensorDataset
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from .multifactordataset import MultifactorDataset

# Fetches a cache ( e.g. some frames from a stream ). Used as a background
# process
def _fetch_cache(datastream, index_queue, data_queue, labels, labels_spk, labels_phone, label_stream_feat, mode, batchsize, shuffle=False):
    while True:
        # Use None as breaking indicator for both queues
        cachesize = index_queue.get()
        if cachesize is None:
            data_queue.put(None)
            break
        try:
            # feats, targets = zip(*[(v, labels[k]) for k, v in (next(datastream)
            #                                            for _ in range(cachesize)) if k in labels])
            feats, labels_id = zip(*[(v, k) for k, v in (next(datastream)
                                                        for _ in range(cachesize)) if k in labels])
            targets = [labels[k] for k in labels_id]
            targets_spk = [labels_spk[k] for k in labels_id]
            targets_phone = [labels_phone[k] for k in labels_id]
            if mode == 3 or mode == 4: 
                targets_feat = [v for k, v in (next(label_stream_feat) 
                                    for _ in range(cachesize)) if k in labels_id]
            else:
                targets_feat = feats
            
        # Finished iterating over the dataset, two cases:
        # 1. Last element has been reached and data is nonempty
        # 2. Last element has been reached and data is empty
        # The second case will be catched in ValueError
        except StopIteration as e:
            # Just return the data
            raise e
        # Returns Valueerror, if zip fails (list is empty, thus no data)
        except ValueError as e:
            data_queue.put(None)
            raise e
        assert feats is not None, "Check the labels!"

        feats = np.concatenate(feats)
        targets = np.concatenate(targets)
        targets_spk = np.concatenate(targets_spk)
        targets_phone = np.concatenate(targets_phone)
        targets_feat = np.concatenate(targets_feat)
        
        
        
        tnetdataset = MultifactorDataset(torch.from_numpy(feats),
                                         torch.from_numpy(targets).long(),
                                         torch.from_numpy(targets_spk).long(),
                                         torch.from_numpy(targets_phone).long(),
                                         torch.from_numpy(targets_feat))

        dataloader = DataLoader(
            tnetdataset, batch_size=batchsize,
            shuffle=shuffle, drop_last=shuffle)
        data_queue.put(dataloader)


class MultifactorStreamDataloader(object):
    """docstring for  KaldiStreamDataloader"""

    def __init__(self, stream, labels, labels_spk, labels_phone, label_stream_feat, countsfile, num_outputs, cachesize=200, peek_fid=None, batchsize=64, mode=0, shuffle=False):
        super(MultifactorStreamDataloader, self).__init__()
        self.stream = stream
        self.labels = labels
        self.labels_spk = labels_spk
        self.labels_phone = labels_phone
        self.label_stream_feat = label_stream_feat
        self.mode = mode

        self.cachesize = cachesize
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.num_outputs = num_outputs
        if isinstance(countsfile, str):
            with open(countsfile) as countsfileiter:
                self.lengths = {k: int(v) for k, v in (
                    l.rstrip('\n').split() for l in countsfileiter)}
        else:
            self.lengths = {k: int(v) for k, v in (
                l.rstrip('\n').split() for l in countsfile)}
        # At least one cache needs to be processed
        self.num_caches = int(
            max(np.ceil(1. * len(self.lengths) / self.cachesize), 1))
        self.cachestartidx = [self.cachesize *
                              i for i in range(self.num_caches)]
        self.nsamples = sum(self.lengths.values())
        if peek_fid:
            # Take the first sample from the data and use it as dim reference
            key, feat = next(kaldi_io.read_mat_ark(self.stream))
            self.inputdim = feat.shape[-1]

    def __iter__(self):
        return itertools.chain.from_iterable(KaldiStreamIter(self))

    def __len__(self):
        return self.num_caches


class KaldiStreamIter(object):
    """
            Stream iterator for Kaldi based features
            This iterator needs the KaldiDataloader as its argument
    """

    def __init__(self, loader):
        super(KaldiStreamIter, self).__init__()
        self.stream = loader.stream
        self.labels = loader.labels
        self.labels_spk = loader.labels_spk
        self.labels_phone = loader.labels_phone
        self.label_stream_feat = loader.label_stream_feat
        self.mode = loader.mode
        self.lengths = loader.lengths
        self.cachesize = loader.cachesize
        self.shuffle = loader.shuffle
        self.batchsize = loader.batchsize
        self.nsamples = loader.nsamples
        self.num_caches = len(loader)
        self.cachestartidx = loader.cachestartidx
        self.idx = 0
        self.startWork()

    def _submitjob(self):
        self.idx += 1
        self.index_queue.put((self.cachesize))

    def startWork(self):
        self.data_queue = multiprocessing.SimpleQueue()
        self.index_queue = multiprocessing.SimpleQueue()
        self.worker = multiprocessing.Process(target=_fetch_cache, args=(kaldi_io.read_mat_ark(
            self.stream), self.index_queue, self.data_queue, self.labels, self.labels_spk, self.labels_phone, kaldi_io.read_mat_ark(self.label_stream_feat), self.mode, self.batchsize, self.shuffle))
        self.worker.start()
        self._submitjob()

    def _shutdown(self):
        self.index_queue.put(None)
        self.worker.join()
        self.worker.terminate()
        # Use -1 as flag for stopiteration
        self.idx = -1

    def __del__(self):
        self._shutdown()

    def __len__(self):
        return self.nsamples

    def __next__(self):
        try:
            # Queue is synchronized, thus will block
            res = self.data_queue.get()
            if self.idx == -1 or not res:
                raise StopIteration
            if self.idx < self.num_caches:
                # Pre-Fetch next item in queue
                self._submitjob()
            elif self.idx >= self.num_caches:
                # The last item - kill the process and queue
                self._shutdown()
            return res
        except KeyboardInterrupt:
            self._shutdown()
            raise StopIteration

    next = __next__  # Python 2 compability

    def __iter__(self):
        return self
