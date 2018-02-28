#!/bin/bash


#SBATCH -o log/decode.%j.log
#SBATCH -e elog/decode.%j.log
#SBATCH -p 3gpuq --gres=gpu:1
#SBATCH --mem=50G


# #SBATCH --nodes=1
# #SBATCH --sockets-per-node=4
# #SBATCH --cores-per-socket=4
# #SBATCH --ntasks-per-core=8



# #SBATCH -w zhangjiagang
# #SBATCH --gres=gpu:1
# #SBATCH -w kunshan



data_dir=data-fbank-64d-25ms-cms-dump-fixpath
train_data="train_si84_multi"
test_data="test_eval92"
exp="exp"
train_ali="tri5c_clean"

test_ali=tri5c_clean
mode=0
modelpath=exp_new/si84-dnn-multifactor/mode$mode/model_best.param

graph=graph_tgpr_5k
latout=dnn_lattice_mode$mode


DELTAS="add-deltas ark:- ark:-"
FEXT="splice-feats --left-context=5 --right-context=5  ark:- ark:-"

INPUTDIM=`echo 64*11*3 | bc`
OUTPUTDIM=`am-info exp/$test_ali/final.mdl | grep 'pdfs' | awk '{print $4}'`
SPK_DIM=`wc -l $data_dir/$train_data/speaker_id | awk '{print $1+1}'`
PHN_DIM=`am-info $exp/$train_ali/final.mdl | grep phones|awk '{print $4}'`
FBK_DIM=`echo "64*11*3" | bc`


LOGPRIOR='prior.dat'

pystring=/mnt/lustre/sjtu/users/jqg01/anaconda3/bin/python 


words=exp/$test_ali/$graph/words.txt
model=exp/$test_ali/final.mdl
graph=exp/"$test_ali"/"$graph"/HCLG.fst

mkdir -p $latout

# num_threads=`expr $nodes \* $threads`
num_threads=8

nj=8
train_data="train_si84_multi"
MEANVAR="$data_dir/$train_data/mean_stde11.dat"

for SGE_TASK_ID in 1 2 3 4 5 6 7 8; do  
TESTDATA="copy-feats scp:$data_dir/$test_data/split${nj}/${SGE_TASK_ID}/feats.scp ark:- | $FEXT |$DELTAS |"
(
echo "python -m py_src.factors_joint.forward --modelpath $modelpath  --logprior $LOGPRIOR --inputdim "$INPUTDIM" --meanvar "$MEANVAR"  --outputdim "$OUTPUTDIM" --spkdim "$SPK_DIM" --phndim "$PHN_DIM" --fbankdim "$FBK_DIM" --mode "$mode" "$TESTDATA" "
python -m py_src.factors_joint.forward --modelpath $modelpath  --logprior $LOGPRIOR --inputdim "$INPUTDIM" --spkdim "$SPK_DIM" --phndim "$PHN_DIM" --fbankdim "$FBK_DIM" --mode "$mode" --meanvar "$MEANVAR" --outputdim "$OUTPUTDIM" "$TESTDATA"  | latgen-faster-mapped-parallel --num-threads=$num_threads --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.0833 --allow-partial=true --word-symbol-table=$words $model $graph ark:- "ark:|gzip -c > $latout/lat.$SGE_TASK_ID.gz" || exit 1;
) & 
done
wait 

