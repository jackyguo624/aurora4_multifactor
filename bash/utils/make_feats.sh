#!/bin/bash

#SBATCH -o log/make-fbank-250ms.%j.log
#SBATCH -e elog/make-fbank-250ms.%j.log
#SBATCH -p cpuq 
#SBATCH --mem=8g



data_dir="data-fbank-64d-250ms"
out_dir="data-fbank-64d-250ms-cms-dump"

# fix wav path



for i in train_si84_multi dev_0330 test_eval92; do
steps/make_fbank.sh --fbank-config conf/fbank64.conf --nj 80 --cmd run.pl $data_dir/$i/
steps/compute_cmvn_stats.sh $data_dir/$i
mkdir -p $out_dir/$i/data
apply-cmvn --utt2spk=ark:$data_dir/$i/utt2spk scp:$data_dir/$i/cmvn.scp scp:$data_dir/$i/feats.scp ark:- | \
    copy-feats --compress=true ark:- ark,scp:$PWD/$out_dir/$i/data/feats.ark,$out_dir/$i/feats.scp
done
