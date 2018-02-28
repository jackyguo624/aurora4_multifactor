#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

#kaldi_dir=/home/souvik/souvik-kaldi-trunk/egs/aurora4/s5/
#[ -f ./$kaldi_dir/path.sh ] && . ./$kaldi_dir/path.sh
# ./$kaldi_dir/cmd.sh
# begin configuration section.
#. ./cmd.sh
cmd=utils/run.pl
stage=0
decode_mbr=true
reverse=false
word_ins_penalty=0.0
min_lmwt=4
max_lmwt=15
#end configuration section.

#[ -f ./path.sh ] && . ./path.sh
#. parse_options.sh || exit 1;

#if [ $# -ne 3 ]; then
#  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
#  echo " Options:"
#  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
#  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
#  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
#  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
#  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
#  echo "    --reverse (true/false)          # score with time reversed features "
#  exit 1;
#fi

catg=$1

dir=$catg
lang_or_graph=$2
#kaldi_dir=/home/souvik/souvik-kaldi-trunk/egs/aurora4/s5/

cat $dir/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/test_filt.txt

symtab=$lang_or_graph/words.txt


# Note: the double level of quoting for the sed command
$cmd LMWT=$min_lmwt:$max_lmwt $dir/log/score.LMWT.log \
   cat $dir/LMWT.tra \| \
    utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' \| \
    compute-wer --text --mode=present \
     ark:$dir/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT || exit 1;

exit 0;
