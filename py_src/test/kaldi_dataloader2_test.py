from .kaldi_dataloader2 import KaldiStreamDataloader as KaldiStreamDataloader2
from .kaldi_dataloader3 import KaldiStreamDataloader as KaldiStreamDataloader3
from ..utils import kaldi_io
from ..utils.kaldi_dataloader import KaldiStreamDataloader
from tqdm import tqdm
import datetime

data_dir="data-fbank-64d-25ms-cms-dump-fixpath"
train_data="train_si84_multi"
train_ali="tri5c_clean"

DELTAS="add-deltas ark:- ark:-"
FEXT="splice-feats --left-context=5 --right-context=5  ark:- ark:-"

stream="copy-feats scp,p:{data_dir}/{train_data}/small_feats.scp ark:- | {FEXT} |{DELTAS} |".format(data_dir=data_dir, train_data=train_data,FEXT=FEXT,DELTAS=DELTAS)
stream2="copy-feats scp,p:{data_dir}/{train_data}2/split{nj}/{i}/feats.scp ark:- | {FEXT} |{DELTAS} |"
labelsf="ali-to-pdf  exp/{train_ali}/final.mdl 'ark:gunzip -c exp/{train_ali}/ali.*.gz |' ark,t:- |".format(train_ali=train_ali)
labels={ k: v for k, v in kaldi_io.read_ali_ark(labelsf)}
countsfile="{data_dir}/{train_data}/small_counts.ark".format(data_dir=data_dir, train_data=train_data)
nj=8

stream2=stream2.format(data_dir=data_dir, train_data=train_data, nj='{nj}',i='{i}', FEXT=FEXT, DELTAS=DELTAS)

splitdataroot="{data_dir}/{train_data}2/split{nj}".format(data_dir=data_dir, train_data=train_data,nj=8)

#print (splitdataroot)
def main():

    count = 0
    st= datetime.datetime.now()

    loader2 = KaldiStreamDataloader2(splitdataroot,labels,1024, batchsize=256)
#    print(loader2.nsamples)
    for x in tqdm(loader2, total=loader2.nsamples // loader2.batchsize):
        data, label  = x
        count+= len(data)
#        if count % 1000 == 0:
#            print(count) 
#    print(count)
    '''
    loader1 = KaldiStreamDataloader(stream,labels,countsfile,1024, batchsize=11, cachesize=3)
    print (loader1.nsamples)
    for x in tqdm(loader1, total=loader1.nsamples // loader1.batchsize):
        data, label  = x
        count+= len(data)
        # print(len(data))
        if count % 10000 == 0:
            print(count) 
    print(count)
    '''

    et = datetime.datetime.now()
    print((et-st).seconds)
if __name__ == "__main__":
    main()

'''

'''

'''
    loader3 = KaldiStreamDataloader3(stream,labels,countsfile,1024, batchsize=256)
    for x in tqdm(loader3, total=loader3.nsamples // loader3.batchsize):
        count+=1
        if count % 10000 == 0:
            print(count) 
'''
