import numpy as np
from ..utils import kaldi_io

cachesize=5
#stream_str='copy-feats scp,o,s:data-fbank-64d-25ms-cms-dump-fixpath/train_si84_multi/feats.scp ark:- |'

#stream_str2='copy-feats scp,o,s:data-fbank-64d-25ms-cms-dump-fixpath/train_si84_clean/feats.scp ark:- |'
stream_str1='pipe1'
stream_str2='pipefile'
datastream=kaldi_io.read_mat_ark(stream_str1)
datastream2=kaldi_io.read_mat_ark(stream_str2)
print(datastream)
print(datastream2)
feats, labels_id = zip(*[(v, k) for k, v in (next(datastream)
                    for _ in range(cachesize))])
print(feats)
print(labels_id)
# targets = [labels[k] for k in labels_id]
# targets_spk = [labels_spk[k] for k in labels_id]
# targets_phone = [labels_phone[k] for k in labels_id]
targets_feat = [ v for k, v in (next(datastream2) 
                    for _ in range(cachesize)) if k in labels_id]


feats = np.concatenate(feats)
targets_feat = np.concatenate(targets_feat)

assert len(feats) == len(targets_feat)
