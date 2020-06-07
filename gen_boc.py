import cluster_dict as cd
from log import log
import os
import numpy as np

from ase.db import connect

log.logger.info('git commit id is: ' + str(os.popen('git --no-pager log -1 --oneline').read()))

clst_tuple_list, total_clst_cnt = cd.get_distance_lst()
total_clst_cnt += 5 # for H C O F N
clst_tuple_list.sort()
log.logger.info('cluster is:'+ str(clst_tuple_list))

ele_clst_idx_dict = {'H':0, 'C':1, 'O':2, 'F':3, 'N':4} # same with atom_dict and atoms_name

def cal_clst_lst_single_atom(atoms, index):
        ret = np.zeros([total_clst_cnt,])
        ret[ele_clst_idx_dict[atoms[index].symbol]] += 1
        for i in range(len(atoms)):
            if(i != index):
                key1 = atoms[i].symbol + atoms[index].symbol
                key2 = atoms[index].symbol + atoms[i].symbol
                min_dis = 9999
                min_idx = -1
                for idx, k in enumerate(clst_tuple_list):
                    if(k[0] == key1 or k[0] == key2):
                        if(atoms.get_distance(i, index) < 5 and # 5 should be same with bond_cluster.py
                                abs(k[1]-atoms.get_distance(i, index)*1000) < min_dis):
                            min_dis = abs(
                                k[1]-atoms.get_distance(i, index)*1000)
                            min_idx = idx
                if(min_idx != -1):
                    # because first 5 elements is single atom
                    ret[min_idx + 5] += 1
        return ret

def cal_clst_lst(atoms):
    ret = np.zeros([total_clst_cnt])
    for i in range(len(atoms)):
        ret += cal_clst_lst_single_atom(atoms, i)
    ret[5:] /= 2
    return ret

qm9_boc_lst = []
# we can check C2H6 result, if it is the same with the BOC paper 
db = connect('./qm9.db')
# 16.1% memory usage of a 128G machine
rows = list(db.select(sort='id'))
for row in rows:
    print(row.id)
    at = row.toatoms()
    boc = cal_clst_lst(at)
    qm9_boc_lst.append(boc)
    # print(type(at.symbols))
    if(str(at.symbols) == 'C2H6'):
        print(boc)

import pickle
with open('qm9_boc_lst.txt', 'wb') as fp:
    pickle.dump(qm9_boc_lst, fp)
