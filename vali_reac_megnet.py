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

with open('megnet_gibbs.lst', 'rb') as fp:
    megnet_gibbs_lst = pickle.load(fp)

print('load megnet finished.')

megnet_gibbs = {}
for (k, v) in megnet_gibbs_lst:
    megnet_gibbs[k] = v

random.shuffle(b)

print('shuffle finished.')

success = 0
failure = 0
neglect = 0
for (id0, id1, id2, direct, g) in b:
    if(abs(g) > 0.05):
        neglect += 1
        continue
    if(megnet_gibbs[id0] + megnet_gibbs[id1] - megnet_gibbs[id2] > 0 and direct < 0):
        success += 1
    elif(megnet_gibbs[id0] + megnet_gibbs[id1] - megnet_gibbs[id2] < 0 and direct > 0):
        success += 1
    else:
        failure += 1
    if((success + failure)%1000000 == 1):
        print('megnet predict accurary: ', success/(success+failure), '; neglect: ', neglect, '; success+failure: ', success+failure)
print('megnet predict accurary: ', success/(success+failure), '; neglect: ', neglect, '; success+failure: ', success+failure)
print('Done.')
