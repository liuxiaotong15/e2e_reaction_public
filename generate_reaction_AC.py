# -*- coding: utf-8 -*

from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
# from tensorflow import keras
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
# logging.info(tf.__version__)
from scipy.optimize import minimize
from itertools import combinations
from ase.visualize import view
from ase.db import connect
import random
from ase.build import sort
from ase import Atom
from ase.io import read, write
from base64 import b64encode, b64decode
import itertools
from log import log

log.logger.info('git commit id is: ' + str(os.popen('git --no-pager log -1 --oneline').read()))

seed = 1234
random.seed(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
log.logger.info('random.seed: '+str(seed))
multi_thd = True

# properties
A = "rotational_constant_A"
B = "rotational_constant_B"
C = "rotational_constant_C"
mu = "dipole_moment"
alpha = "isotropic_polarizability"
homo = "homo"
lumo = "lumo"
gap = "gap"
r2 = "electronic_spatial_extent"
zpve = "zpve"
U0 = "energy_U0"
U = "energy_U"
H = "enthalpy_H"
G = "free_energy"
Cv = "heat_capacity"

db = connect('./qm9.db')
# 16.1% memory usage of a 128G machine
rows = list(db.select(sort='id'))
# rows = list(db.select('id<200'))

atom_names = ['H', 'C', 'O', 'F', 'N']
atom_dict = {'H': 0, 'C':1, 'O':2, 'F':3, 'N':4}
atom_cnt_lst = []
atom_cnt_dict = {}

def get_data_pp(idx, type):
    # extract properties
    prop = 'None'
    # row = db.get(id=idx)
    row = rows[idx-1]
    if(row.id != idx):
        1/0
    # extract from schnetpack source code
    shape = row.data["_shape_" + type]
    dtype = row.data["_dtype_" + type]
    prop = np.frombuffer(b64decode(row.data[type]), dtype=dtype)
    prop = prop.reshape(shape)
    # log.logger.info('idx = ' + str(idx) + ', ' + str(row.toatoms().symbols) + ', ' + str(prop[0]))
    return prop

def hash_1d_array(arr):
    ret = 0
    arr = list(arr)
    for i in range(len(arr)):
        ret += (arr[i] * pow(50, (len(arr) - i)))
    return ret

def multi_thd_reac(reac_lst_orig):
    g1 = get_data_pp(idx=reac_lst_orig[0], type=G)[0]
    g2 = get_data_pp(idx=reac_lst_orig[1], type=G)[0]
    
    if(g1 > g2): # A => C
        ret = [reac_lst_orig[0], reac_lst_orig[1]]
    else:
        ret = [reac_lst_orig[1], reac_lst_orig[0]]
    return ret

if __name__ == '__main__':
    for row in rows:
        at = row.toatoms()
        atom_cnt = [0] * len(atom_names)
        for a in at:
            atom_cnt[atom_dict[a.symbol]] += 1
        atom_cnt_lst.append(np.array(atom_cnt))
        k = hash_1d_array(np.array(atom_cnt))
        if(atom_cnt_dict.__contains__(k)):
            atom_cnt_dict[k].append(row.id)
        else:
            atom_cnt_dict[k] = [row.id]
    print(len(rows)) # 133885
    print(len(atom_cnt_lst)) # 133885
    print(len(atom_cnt_dict)) # 621

    dk = list(atom_cnt_dict.keys())
    random.shuffle(dk)
    max_molcl_cnts = 0
    for k in dk:
        max_molcl_cnts = max(max_molcl_cnts, len(atom_cnt_dict[k]))
    print('max molecules in group is:', max_molcl_cnts)
    reac_cnt = 0
    reac_lst = []
    # so the row format in the reaction database should be like:
    # id_G_bigger, id_G_smaller store in numpy and savetxt

    test_group_count = 20
    for i in range(len(dk) - test_group_count):
        reac_cnt += (len(atom_cnt_dict[dk[i]])*(len(atom_cnt_dict[dk[i]])-1)/2)
        for j in range(len(atom_cnt_dict[dk[i]])):
            for k in range(j+1, len(atom_cnt_dict[dk[i]])):
                reac_lst.append([atom_cnt_dict[dk[i]][j], atom_cnt_dict[dk[i]][k]])
    
    log.logger.info('len(reac_lst): ' + str(len(reac_lst)))
    log.logger.info('reac_cnt: ' + str(reac_cnt))

    if(multi_thd):
        pool = multiprocessing.Pool(3)
        reac_lst_1 = pool.map(multi_thd_reac, reac_lst)
        reac_lst = reac_lst_1
    else:
        reac_lst_1 = []
        for i in range(len(reac_lst)):
            reac_lst_1.append(multi_thd_reac(reac_lst[i]))
            reac_lst = reac_lst_1
    import pickle
    with open('reactions_AC.txt', 'wb') as fp:
        pickle.dump(reac_lst, fp)

    # save test group case
    # (id1, id2) also (id2, id1) later, all id are row id.
    
    reac_cnt = 0
    reac_lst = []
    for i in range(len(dk) - test_group_count, len(dk)):
        reac_cnt += (len(atom_cnt_dict[dk[i]])*(len(atom_cnt_dict[dk[i]])-1))
        for j in range(len(atom_cnt_dict[dk[i]])):
            for k in range(len(atom_cnt_dict[dk[i]])):
                if(j != k):
                    reac_lst.append([atom_cnt_dict[dk[i]][j], atom_cnt_dict[dk[i]][k]])
    
    log.logger.info('len(reac_lst): ' + str(len(reac_lst)))
    log.logger.info('reac_cnt: ' + str(reac_cnt))
    
    # so the sum of the reaction should be reaction_1 + reaction_2/2
    import pickle
    with open('reactions_AC_test_group.txt', 'wb') as fp:
        pickle.dump(reac_lst, fp)
