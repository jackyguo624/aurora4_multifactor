from tqdm import tqdm
import itertools

cachesize=10
r= {i: i+1 for i in range(1,4)}
it=iter([r, r, r])

res =[x for x in (next(it) for _ in range(cachesize))]
print (res)


res =[x for x in (next(it) for _ in range(cachesize))]
print (res)
'''
r= {i: i+1 for i in range(1,4)}
it1=iter([r, r, r])

it2=iter([r, r, r])

itr3=iter([r, r, r])

itertools.chain.from_iterable((iter))
'''
