#!/bin/bash


#SBATCH -o log/score.%j.log
#SBATCH -e elog/score.%j.log
#SBATCH -p cpuq/sjtu
#SBATCH --mem=10G

#SBATCH --nodes=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --ntasks-per-core=8




echo 16 > $1/num_jobs

#lm=bg4_5k
lm=tgpr_5k

dir=$1
graphdir=exp/tri5c_clean/graph_${lm}
data_dir=data-fbank-64d-25ms-cms-dump-fixpath/test_eval92/
#data_dir=data-fbank-64d-25ms-cms-dump-fixpath/dev_0330/

local/score.sh  --min_lmwt 4 --max_lmwt 15  $data_dir $graphdir $1

#./steps/score_kaldi.sh $data_dir $graphdir $1
