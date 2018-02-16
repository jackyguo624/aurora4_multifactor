data_path="data-fbank-64d-250ms-cms-dump"




# make small set for test src
for x in train_si84_multi dev_0330 test_eval92; do
head -n 10 $data_path/$x/feats.scp > $data_path/$x/small_feats.scp
done

# make count file for iterator in python
for x in train_si84_multi dev_0330 test_eval92; do
feat-to-len scp:$data_path/$x/feats.scp ark,t:$data_path/$x/counts.ark
feat-to-len scp:$data_path/$x/small_feats.scp ark,t:$data_path/$x/small_counts.ark
done
