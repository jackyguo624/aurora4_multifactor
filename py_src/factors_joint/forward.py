import sys
import torch
import argparse
from .DNN import DNN
from ..utils import kaldi_io
from torch.autograd import Variable
import numpy as np

def kaldi_mat_ark_stream(inp):
    inp = inp.rstrip(' ')
    if not inp[-1] == "|":
        raise argparse.ArgumentTypeError("This inputs needs to be a stream")
    return kaldi_io.read_mat_ark(inp)

def load_mean_var(fpath):
    mv = np.loadtxt(fpath)
    return torch.FloatTensor(mv[0]), torch.FloatTensor(mv[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feats', type=kaldi_mat_ark_stream,
                        help="Stream, e.g. copy-feats scp:feats ark:- |")
    parser.add_argument('output', default=sys.stdout.buffer,
                        type=argparse.FileType('wb'), nargs="?")
    parser.add_argument('--meanvar', type=str)

    # inputdim & outputdim, help to init network
    parser.add_argument("--inputdim", type=int, required=True)
    parser.add_argument("--spkdim", type=int, required=True)
    parser.add_argument("--phndim", type=int, required=True)
    parser.add_argument("--fbankdim", type=int, required=True)
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument("--outputdim", type=int, required=True)

    parser.add_argument('--modelpath', type=str)
    parser.add_argument('--logprior', type=argparse.FileType('r'))
    parser.add_argument('--nocuda', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    model_path = args.modelpath
    net = DNN(args.inputdim, args.outputdim, args.spkdim, args.phndim, args.fbankdim,
                mode=args.mode)
    net.load_state_dict(torch.load(model_path))
    #net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict'])
    log_prior = np.loadtxt(args.logprior)
    if args.meanvar:
        tr_mean, tr_var = load_mean_var(args.meanvar)
    net.eval()
    if not args.nocuda:
        net.cuda()
    for k, v in args.feats:
        v = torch.from_numpy(v)
        if args.meanvar:
            v = torch.add(v, -tr_mean)
            v = torch.div(v, tr_var)
        if not args.nocuda:
            v = v.cuda()
        v = Variable(v, volatile=True)
        post, _, _, _, _ = net(v)
        post = post.cpu().data.numpy()
        post = post - log_prior
        kaldi_io.write_mat(args.output, post, k)
    
if __name__ == "__main__":
    main()
