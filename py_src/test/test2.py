#!/usr/bin/env python
import sys, os, re, gzip, struct

# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, mode="rb"):
  if not isinstance(cmd, str):
    raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

  import subprocess, io, threading

  # cleanup function for subprocesses,
  def cleanup(proc, cmd):
    ret = proc.wait()
    if ret > 0:
      raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
    return

  # text-mode,
  if mode == "r":
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdout)
  elif mode == "w":
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdin)
  # binary,
  elif mode == "rb":
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdout
  elif mode == "wb":
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdin
  # sanity,
  else:
    raise ValueError("invalid mode %s" % mode)


file1='copy-feats scp,o,s:data-fbank-64d-25ms-cms-dump-fixpath/train_si84_multi/feats.scp ark:- |'
file2='copy-feats scp,o,s:data-fbank-64d-25ms-cms-dump-fixpath/train_si84_clean/feats.scp ark:- |'


fd1 = popen(file1[:-1], 'rb') 
fd2 = open('./pipe1', 'rb')

print(fd1.read(1))
print(fd2.read(1))
print(fd1.read(1))
print(fd2.read(1))
print(fd1.read(1))
print(fd2.read(1))

fd1.close()
fd2.close()





