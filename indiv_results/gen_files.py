#!/usr/bin/python



import sys,re,os,os.path


if len(sys.argv) != 5:
    print( 'python gen_files.py <source_txt> <tra-flist> <A/B/C/D> <output_dir> ')
    sys.exit(1)

src_fname = sys.argv[1]
tra_fname = sys.argv[2]
opt = sys.argv[3]
out_dir = sys.argv[4]


cat_list = None
if opt == 'A':
    cat_list = ['0']
elif opt == 'B':
    cat_list = ['1','2','3','4','5','6']
elif opt == 'C':
    cat_list = ['7']
else:
    cat_list = ['8','9','a','b','c','d']

src_map = {}
fp = open(src_fname,'r')
while True:
    data = fp.readline()
    if data == '':
        break
    data = data.strip()
    sp = re.split('\s+',data)
    fid = sp.pop(0)
    src_map[fid] = data
fp.close()


def get_list(fname):
    c = []
    fp = open(fname,'r')
    while True:
        data = fp.readline()
        if data == '':
            break
        data = data.strip()
        c.append(data)
    fp.close()
    return c

flist = get_list(tra_fname)


fpw1 = open(os.path.join(out_dir,'text'),'w')
for i in range(0,len(flist)):

    tra_path = flist[i]
    tra_fname = os.path.basename(tra_path)
    fp = open(tra_path,'r')
    fpw2 = open(os.path.join(out_dir,tra_fname),'w')
    while True:
        data = fp.readline()
        if data == '':
            break
        data = data.strip()
        sp = re.split('\s+',data)
        fid = sp[0]
        if fid[-1] in cat_list:
            if i == 0:
                fpw1.write(src_map[fid]+'\n')
            fpw2.write(data+'\n')
    fpw2.close()
fpw1.close()
