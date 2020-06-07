import pickle
# with open('reactions.txt', 'wb') as fp:
#     pickle.dump(reac_lst, fp)

import random

seed = 1234
random.seed(seed)
# np.random.seed(seed)

with open('reactions_AC.txt', 'rb') as fp:
    b = pickle.load(fp)
print(b[81040095])
print(len(b)) # this should be same with the output in all.log

random.shuffle(b)

with open('qm9_boc_lst.txt', 'rb') as fp:
    c = pickle.load(fp)

ret = []
pos_cnt = 10000000
neg_cnt = 10000000
i = 0
for i in range(pos_cnt):
    tmp = []
    tmp.append(c[b[i][0]-1]-c[b[i][1]-1])
    tmp.append(1)
    ret.append(tmp)
for i in range(neg_cnt):
    tmp = []
    tmp.append(c[b[i][1]-1]-c[b[i][0]-1])
    tmp.append(-1)
    ret.append(tmp)
   
with open('dataset.lst', 'wb') as fp:
    pickle.dump(ret, fp)

del b
del ret
import gc
gc.collect()

with open('reactions_AC_test_group.txt', 'rb') as fp:
    b = pickle.load(fp)

print(len(b))
ret = []
for b_idx in range(len(b)):
    tmp = []
    tmp.append(b[b_idx][0])
    tmp.append(b[b_idx][1])
    tmp.append(c[b[b_idx][0]-1]-c[b[b_idx][1]-1])
    ret.append(tmp)

with open('test_dataset.lst', 'wb') as fp:
    pickle.dump(ret, fp)
