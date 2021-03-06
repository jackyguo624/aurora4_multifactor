
module load kaldi

tt_path="data-fbank-64d-25ms-cms-dump"
fix_path="data-fbank-64d-25ms-cms-dump-fixpath"

# fix feats
for x in train_si84_multi dev_0330 test_eval92; do
mkdir -p $fix_path/$x
cat $tt_path/$x/feats.scp | awk '{print $1 " /mnt" $2}' > $fix_path/$x/feats.scp
done

# make small set for test src
for x in train_si84_multi dev_0330 test_eval92; do
head -n 10 $fix_path/$x/feats.scp > $fix_path/$x/small_feats.scp
done

# make count file for iterator in python
for x in train_si84_multi dev_0330 test_eval92; do
feat-to-len scp:$fix_path/$x/feats.scp ark,t:$fix_path/$x/counts.ark
feat-to-len scp:$fix_path/$x/small_feats.scp ark,t:$fix_path/$x/small_counts.ark
done
