import argparse

tr_countfile="/mnt/lustre/sjtu/users/jqg01/asr/aurora4/data-fbank-64d-25ms-cms-dump-fixpath/train_si84_multi/counts.ark"

train_skr_ali="/mnt/lustre/sjtu/users/jqg01/asr/aurora4/data-fbank-64d-25ms-cms-dump-fixpath/train_si84_multi/spk_ali.ark"

name_map="/mnt/lustre/sjtu/users/jqg01/asr/aurora4/data-fbank-64d-25ms-cms-dump-fixpath/train_si84_multi/speaker_id"


name={}

def main():
    tr_out = open(train_skr_ali, "w")
    with open(tr_countfile) as countsfileiter:
        lengths = {k: int(v) for k, v in (l.rstrip('\n').split() for l in countsfileiter)}
    for k, v in lengths.items():
        if k[:3] not in name:
            _id = len(name)
            name[k[:3]]=_id
        else:
            _id = name[k[:3]]
        tr_out.write(" ".join([k, " ".join([str(_id) for i in range(v)])])+"\n")
    tr_out.close()
    # dump name mapping
    name_out = open(name_map, "w")
    for k, v in name.items():
        name_out.write(" ".join([k, str(v)])+"\n")
    name_out.close()

if __name__ == '__main__':
    main()
'''
dev_countfile="/mnt/lustre/sjtu/users/jqg01/asr/aurora4/data-fbank-64d-25ms-cms-dump-fixpath/dev_0330/counts.ark"
dev_skr_ali="/mnt/lustre/sjtu/users/jqg01/asr/aurora4/data-fbank-64d-25ms-cms-dump-fixpath/dev_0330/spk_ali.ark"
    dev_out = open(dev_skr_ali, "w")
    with open(dev_countfile) as countsfileiter:
        lengths = {k: int(v) for k, v in (l.rstrip('\n').split() for l in countsfileiter)}
    for k, v in lengths.items():
        if k[:3] not in name:
            raise ValueError("speak {} is not in train set speaker".format(k[:3]))
        _id = name[k[:3]]
        dev_out.write(" ".join([k, " ".join([str(_id) for i in range(v)])]))
    dev_out.close()
'''
