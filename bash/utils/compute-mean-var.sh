#!/bin/bash

#SBATCH --mem=5G
#SBATCH --output=slurm_mean_var_%j.out
#SBATCH -p cpuq

data_dir="data-fbank-64d-25ms-cms-dump-fixpath"
# data_dir="data-fbank-64d-250ms-cms-dump"
train_data="train_si84_multi"

DELTAS="add-deltas ark:- ark:-"
FEXT="splice-feats --left-context=5 --right-context=5  ark:- ark:-"

traindata="copy-feats scp:$data_dir/$train_data/feats.scp ark: | $FEXT | $DELTAS  |"
traincount="$data_dir/$train_data/counts.ark"
output="$data_dir/$train_data/mean_stde11.dat"
/mnt/lustre/sjtu/users/jqg01/anaconda3/bin/python -m py_src.utils.compute_mean_var "$traindata" "$traincount" "$output"
#/cm/shared/apps/python/3.5.4/bin/python3.5 -m py_src.utils.compute_mean_var "$traindata" "$traincount" "$output"
