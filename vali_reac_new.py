import pickle
import h5py
import random
import os
import numpy as np
seed = 1234
random.seed(seed)
# np.random.seed(seed)

with open('reactions.txt', 'rb') as fp:
    b = pickle.load(fp)
print(b[81040095])
print(len(b)) # this should be same with the output in all.log

random.shuffle(b)
# one_cnt = 0
# for item in b:
#     if(item[3] > 0):
#         one_cnt += 1
# print(one_cnt)

with open('qm9_id_boc.lst', 'rb') as fp:
    c = pickle.load(fp)

print(c[100000])
print(len(c))


i = 0
for idx in range(100):
    ret = []
    pos_cnt = 500000
    neg_cnt = 500000
    while((pos_cnt > 0 or neg_cnt > 0) and i < len(b)):
        # if(abs(b[i][4]) > 0.05):
        #     i += 1
        #     continue
        # tmp = []
        # need db idx and list idx conversion
        if(c[b[i][0]-1][0] != b[i][0] or c[b[i][1]-1][0] != b[i][1] or c[b[i][2]-1][0] != b[i][2]):
            1/0
        tmp = (c[b[i][0]-1][1]+c[b[i][1]-1][1]-c[b[i][2]-1][1])
        # print(tmp.shape)
        tmp = np.concatenate((tmp, np.array([b[i][3]])))
        # print(tmp.shape)
        if(pos_cnt > 0 and b[i][3] > 0):
            ret.append(tmp)
            pos_cnt -= 1
        elif(neg_cnt > 0 and b[i][3] < 0):
            ret.append(tmp)
            neg_cnt -= 1
        else:
            pass
        if pos_cnt == 0 and neg_cnt == 0:
            random.shuffle(ret)
            os.system('rm -rf ' + 'dataset_new_' + str(idx+1) + '.hdf5')
            f = h5py.File('dataset_new_' + str(idx+1) + '.hdf5', 'w')
            f.create_group('/grp1') # or f.create_group('grp1')
            f.create_dataset('dset1', compression='gzip', data=np.array(ret)) # or f.create_dataset('/dset1', data=data)
            f.close()
            # with open('dataset_new_' + str(idx+1) + '.lst', 'wb') as fp:
            #     pickle.dump(ret, fp)
            ret = []
        i += 1

print('total reaction count: ', i)
print('Done.')
