import numpy as np
from tqdm import tqdm
from .kaldi_dataloader import KaldiStreamDataloader
import argparse
import logging

logger = logging.getLogger('compute_mean_var')
logging.basicConfig()
logger.setLevel(logging.DEBUG)

def mean(dataloader,batchsize):
    _sum = None
    _count = 0
    for data_label in tqdm(dataloader,total=dataloader.nsamples*1./batchsize):
        v,l = data_label
        v = v.numpy()
        _sum = np.zeros(v.shape[1]) if _sum is None else _sum
        _count += v.shape[0]
        _sum = _sum + v.sum(axis=0)
    return _sum / _count


def stderr(dataloader, means, batchsize):
    _sum = None
    _count = 0
    for data_label in tqdm(dataloader, total=dataloader.nsamples*1./batchsize):
        v, l = data_label
        v = v.numpy()
        _sum = np.zeros(v.shape[1]) if _sum is None else _sum
        _count += v.shape[0]
        _sum += np.square(v - means).sum(axis=0)
    if _count - 1 <=0:
        return None
    _var =  _sum / (_count - 1)
    return np.sqrt(_var)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('traincounts', type=argparse.FileType('r'))
    parser.add_argument('output', type=str)
    batchsize = 50

    args = parser.parse_args()


    dataloader = KaldiStreamDataloader(stream=args.input, labels=None, num_outputs=0,# no need ouput
                                       countsfile=args.traincounts, batchsize=batchsize, cachesize=batchsize*1, shuffle=False)


    means = mean(dataloader, batchsize)

    stderrs = stderr(dataloader, means, batchsize)

    np.savetxt(args.output,(means,stderrs),header="first line:means,second line:stderrs", fmt='%.8f')

if __name__ == '__main__':
    main()
