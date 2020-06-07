import pickle
# with open('reactions.txt', 'wb') as fp:
#     pickle.dump(reac_lst, fp)

import random

seed = 1234
random.seed(seed)
# np.random.seed(seed)

with open('reactions.txt', 'rb') as fp:
    b = pickle.load(fp)
print(b[81040095])
print(len(b)) # this should be same with the output in all.log

random.shuffle(b)
one_cnt = 0
for item in b:
    if(item[3] > 0):
        one_cnt += 1
print(one_cnt)

with open('qm9_boc_lst.txt', 'rb') as fp:
    c = pickle.load(fp)

ret = []
pos_cnt = 5000000
neg_cnt = 5000000
i = 0
# for i in range(10000000):
while((pos_cnt > 0 or neg_cnt > 0) and i < len(b)):
    if(abs(b[i][4]) > 0.05):
        i += 1
        continue
    tmp = []
    # need db idx and list idx conversion
    tmp.append(c[b[i][0]-1]+c[b[i][1]-1]-c[b[i][2]-1])
    tmp.append(b[i][3])
    if(pos_cnt > 0 and b[i][3] > 0):
        ret.append(tmp)
        pos_cnt -= 1
    elif(neg_cnt > 0 and b[i][3] < 0):
        ret.append(tmp)
        neg_cnt -= 1
    else:
        pass
    i += 1

print('total reaction count: ', len(ret))

with open('dataset.lst', 'wb') as fp:
    pickle.dump(ret, fp)
# print(c[8105]+c[810]-c[18105])
print('Done.')
