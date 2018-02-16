from ..utils import kaldi_io
import multiprocessing, logging
#fh=logging.FileHandler('./filehandler.log',mode='w')
#logger=multiprocessing.log_to_stderr()
#logger.addHandler(fh)

#logger.error("this is an error")

filehandler = logging.FileHandler('/mnt/lustre/sjtu/users/jqg01/asr/aurora4/readprocessing.log',mode='w')
logger=multiprocessing.log_to_stderr()
logger.addHandler(filehandler)
logger.setLevel(logging.DEBUG)


def touchlog(meg):
    logger.error(meg)

meg="subprocess error"
p1 = multiprocessing.Process(target=touchlog,args=(meg,))
p1.start()

p1.join()

p1.terminate()

