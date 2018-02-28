#!/bin/bash

. ./cmd.sh
. ./path.sh

lm=bg4_5k

. parse_options.sh || exit 1;

if [ ! $# -eq 1 ]; then
    echo "Usage: $0 decodedir"
    echo "Example: $0 exp_cntk/resnet/"
    exit 1;
fi

dir=$1
textfile=$dir/scoring_kaldi/test_filt.txt
graphdir=exp/tri5c_clean/graph_${lm}

rm $dir/decode_${lm}_eval92/tra.flist

mkdir -p $dir/decode_${lm}_eval92 

for x in $dir/scoring/*.tra; do echo $x >> $dir/decode_${lm}_eval92/tra.flist; done

for x in A B C D; do mkdir -p $dir/decode_${lm}_eval92$x; done

for x in A B C D; do python indiv_results/gen_files.py data-fbank-64d-25ms-cms-dump-fixpath/test_eval92/text $dir/decode_${lm}_eval92/tra.flist $x $dir/decode_${lm}_eval92$x; done

for x in A B C D; do indiv_results/score.sh $dir/decode_${lm}_eval92$x $graphdir; done


echo
echo "-----------------------------------------------------"
#/decode_${lm}_eval92

w=`grep WER $dir/wer_* | utils/best_wer.sh | awk -F_ '{print $NF}'`
echo $dir/wer_$w && grep WER $dir/wer_${w}

for x in $dir; do w=`[ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh | awk -F_ '{print $NF}'`; for y in A B C D; do for z in $x/decode_${lm}_eval92${y}; do [ -d $z ] && echo $z/wer_$w && grep WER $z/wer_${w}; done; done; echo ; echo "-----------------------------------------------------"; echo ;done | cat

