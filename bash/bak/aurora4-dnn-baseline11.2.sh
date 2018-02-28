#!/bin/bash

#SBATCH -o log/aurora4-dnnbaseline11_std_p.%j.log
#SBATCH -e elog/aurora4-dnnbaseline11_std_p.%j.log
#SBATCH -p 3gpuq --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --ntasks-per-core=8
#SBATCH --sockets-per-node=2 
#SBATCH --cores-per-socket=2
#SBATCH -w zhangjiagang

# #SBATCH -w cherry



data_dir="data-fbank-64d-25ms-cms-dump-fixpath"
# data_dir="data-fbank-64d-250ms-cms-dump"
train_data="train_si84_multi2"
dev_data="dev_0330_2"


exp="exp"

# train_ali="tri5c_multi_ali_si84"
# dev_ali="tri5c_multi_ali_dev_0330"

train_ali="tri5c_clean"
dev_ali="tri5c_clean_ali_dev_0330_clean"


TRAINDATA="$data_dir/$train_data/split8"
CVDATA="$data_dir/${dev_data}/split8"

TRAINLABEL="ali-to-pdf  $exp/$train_ali/final.mdl 'ark:gunzip -c $exp/$train_ali/ali.*.gz |' ark,t:- |"
CVLABEL="ali-to-pdf  $exp/$dev_ali/final.mdl 'ark:gunzip -c $exp/$dev_ali/ali.*.gz |' ark,t:- |"

INPUTDIM=`echo "64*11*3" | bc` 

OUTPUTDIM=`am-info $exp/$train_ali/final.mdl | grep 'pdfs' | awk '{print $4}'`

MEANVAR="$data_dir/$train_data/mean_stde11.dat"

OUTPUT="exp_new/si84-dnn-baseline11_std_parallel"

PRETRAIN_DIR="exp_new/si84-dnn-baseline11_std_parallel/checkpoint.th"

echo -n "#"

#--pretraindir "$PRETRAIN_DIR"
/mnt/lustre/sjtu/users/jqg01/anaconda3/bin/python -m py_src.dnnbaseline.train2 "$TRAINDATA" "$TRAINLABEL" "$CVDATA" "$CVLABEL" \
"$INPUTDIM" "$OUTPUTDIM" "$MEANVAR"  -lr 1e-3 -e 20  -o "$OUTPUT"

