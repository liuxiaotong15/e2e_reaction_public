import math
from itertools import combinations
from log import log
import os
import numpy as np

from ase.db import connect

import pickle
with open('clst_dict.dct', 'rb') as fp:
    clst_dict = pickle.load(fp)

clst_tuple_list = []
for k in clst_dict:
    v_lst = [v for v in clst_dict[k]]
    for v in v_lst:
        clst_tuple_list.append((k,list(v)))
clst_tuple_list = sorted(clst_tuple_list)

# print(clst_tuple_list)
print('BoC contain ', len(clst_tuple_list), ' elements.')
atom_names = ['H', 'C', 'O', 'N']
atom_dict = {'H': 0, 'C':1, 'O':2, 'N':3}

clst_atom_cnt_min = 2
clst_atom_cnt_max = 4
max_clst_diameter = 3.5 # A

def cal_clst_lst(atoms):
    ret = np.zeros(len(clst_tuple_list))
    for atom_cnts in range(clst_atom_cnt_min, clst_atom_cnt_max+1):
        for idx_tup in list(combinations(range(len(atoms)), atom_cnts)):
            # print('idx_tup: ', idx_tup)
            clst_vaild = True
            clst_hash = []
            for idx_pair in list(combinations(list(idx_tup), 2)):
                # print('idx_pair: ', idx_pair)
                d = atoms.get_distance(idx_pair[0], idx_pair[1])
                if(d > max_clst_diameter):
                    clst_vaild =False
                    break
                # d = 1-1/(d**2) * 1000
                # d = math.log(d-0.95) * 1000
                clst_hash.append((''.join(sorted(atoms[idx_pair[0]].symbol+atoms[idx_pair[1]].symbol)), d))
            if clst_vaild:
                clst_hash = sorted(clst_hash)
                # print('clst_hash: ', clst_hash)
                dd_key = ''.join(sorted(list(atoms[idx].symbol for idx in idx_tup)))
                # print('dd_key: ', dd_key)
                dd_value = [tup[1] for tup in clst_hash]
                # print('dd_value: ', dd_value)
                best_index = -1
                similar_score = 9999999999
                for i in range(len(clst_tuple_list)):
                    if dd_key != clst_tuple_list[i][0]:
                        continue
                    else:
                        ss_tmp = 0
                        for j in range(len(clst_tuple_list[i][1])):
                            ss_tmp += (clst_tuple_list[i][1][j] - dd_value[j])**2
                        if ss_tmp < similar_score:
                            similar_score = ss_tmp
                            best_index = i
                if best_index != -1:
                    ret[best_index] += 1

    return ret

qm9_boc_lst = []
# we can check C2H6 result, if it is the same with the BOC paper 
db = connect('./qm9.db')
# 16.1% memory usage of a 128G machine
rows = list(db.select(sort='id'))

# for row in rows:
def multi_thd_func(row):
    if row.id % 1000 == 1:
        print(row.id)
    at = row.toatoms()
    boc = cal_clst_lst(at)
    if(str(at.symbols) == 'C2H6'):
        print('C2H6 BoC is: ', boc)
        print('length of BoC is: ', len(boc))
        for i in range(len(boc)):
            if boc[i] != 0:
                print(clst_tuple_list[i][0], boc[i])
    return (row.id, boc)

import multiprocessing
pool = multiprocessing.Pool(24)
qm9_boc_lst = pool.map(multi_thd_func, rows)

import pickle
with open('qm9_id_boc.lst', 'wb') as fp:
    pickle.dump(qm9_boc_lst, fp)

print('total stored BoC in qm9_boc_lst.txt is: ', len(qm9_boc_lst))