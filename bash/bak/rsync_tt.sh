source_dir=/mnt/speechlab/users/tt123/asr/aurora4_new/data-fbank-64d
target_dir=data-fbank-64d-250ms

for i in train_si84_multi dev_0330 test_eval92;do
echo "rsync -av  $source_dir/$i $target_dir/ --exclude=$source_dir/$i/data"
rsync -av  $source_dir/$i $target_dir/ --exclude=$tartget_dir/$i/data --exclude=*.log
done 

for i in train_si84_multi dev_0330 test_eval92;do
mv $target_dir/$i/wav.scp $target_dir/$i/.wav.scp.bak 
awk '{print $1 " /mnt" $2}' $target_dir/$i/.wav.scp.bak > $target_dir/$i/wav.scp
done
