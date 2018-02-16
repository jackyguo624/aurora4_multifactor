#!/bin/bash

#SBATCH -o log/aurora4-dnnbaseline.%j.log
#SBATCH -e elog/aurora4-dnnbaseline.%j.log
#SBATCH -p gpuq/sjtu --gres=gpu:1
#SBATCH --mem=20G
# #SBATCH --sockets-per-node=2 
# #SBATCH --cores-per-socket=10
# #SBATCH --threads-per-core=2
#SBATCH -w cherry


data_dir="data-fbank-64d-25ms-cms-dump-fixpath"
# data_dir="data-fbank-64d-250ms-cms-dump"
train_data="train_si84_multi"
dev_data="dev_0330"
# use_small="small_"

exp="exp"

# train_ali="tri5c_multi_ali_si84"
# dev_ali="tri5c_multi_ali_dev_0330"

train_ali="tri5c_clean"
dev_ali="tri5c_clean_ali_dev_0330_clean"


DELTAS="add-deltas ark:- ark:-"
FEXT="splice-feats --left-context=11 --right-context=11  ark:- ark:-"

TRAINDATA="copy-feats scp:$data_dir/$train_data/${use_small}feats.scp ark:- | $FEXT |$DELTAS |"
CVDATA="copy-feats scp:$data_dir/${dev_data}/${use_small}feats.scp ark:- | $FEXT |$DELTAS |"

TRAINLABEL="ali-to-pdf  $exp/$train_ali/final.mdl 'ark:gunzip -c $exp/$train_ali/ali.*.gz |' ark,t:- |"
CVLABEL="ali-to-pdf  $exp/$dev_ali/final.mdl 'ark:gunzip -c $exp/$dev_ali/ali.*.gz |' ark,t:- |"

TRAINCOUNTS=$data_dir/$train_data/${use_small}counts.ark
CVCOUNTS=$data_dir/$dev_data/${use_small}counts.ark
INPUTDIM=`echo "64*23*3" | bc` 
OUTPUTDIM=`am-info $exp/$train_ali/final.mdl | grep 'pdfs' | awk '{print $4}'`

MEANVAR="$data_dir/$train_data/mean_var.dat"

OUTPUT="exp_new/si84-dnn-baseline"

PRETRAIN_DIR="exp_new/si84-dnn-baseline/checkpoint.th"

echo -n "#"
cat <<EOF
/home/sjtu/jqg01/anaconda3/bin/python -m py_src.dnnbaseline.train "$TRAINDATA" "$TRAINLABEL" "$CVDATA" "$CVLABEL" "$TRAINCOUNTS" "$CVCOUNTS" \
    "$OUTPUTDIM" "$MEANVAR" -lr 1e-2 -o "$OUTPUT"
EOF

/mnt/lustre/sjtu/users/jqg01/anaconda3/bin/python -m py_src.dnnbaseline.train "$TRAINDATA" "$TRAINLABEL" "$CVDATA" "$CVLABEL" "$TRAINCOUNTS" "$CVCOUNTS" \
   "$INPUTDIM" "$OUTPUTDIM" "$MEANVAR" --pretraindir "$PRETRAIN_DIR"  -lr 1e-2 -e 100 -o "$OUTPUT"
