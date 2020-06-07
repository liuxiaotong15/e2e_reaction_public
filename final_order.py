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

import pickle
with open('competition_with_ret_0.txt', 'rb') as fp:
    id_with_ret = pickle.load(fp)
with open('qm9_gibbs_order.txt', 'rb') as fp:
    qm9_gibbs_order = pickle.load(fp)
with open('megnet_gibbs.lst', 'rb') as fp:
    megnet_gibbs_lst = pickle.load(fp)

megnet_gibbs = {}
for (k, v) in megnet_gibbs_lst:
    megnet_gibbs[k] = v

print(qm9_gibbs_order[0])
last_formula = ''
id_cnt_dict = {}
id_megnet_gibbs_dict = {}


predict_actual_order_list = []
megnet_predict_actual_order_list = []
for i in range(len(id_with_ret)):
    # print(id_with_ret[i][3])
    # print(rows[id_with_ret[i][0]-1].formula, rows[id_with_ret[i][1]-1].formula)
    if(last_formula != rows[id_with_ret[i][0]-1].formula):
        # sort a dict by value
        id_cnt_dict = {k:v for k,v in sorted(id_cnt_dict.items(), key=lambda item: item[1])}
        print('Prediction: id, total loss cnt: ')
        print(id_cnt_dict)
        
        id_megnet_gibbs_dict = {k:v for k,v in sorted(id_megnet_gibbs_dict.items(), key=lambda item: item[1])}
        print('megnet Prediction: id, total loss cnt: ')
        print( id_megnet_gibbs_dict)

        id_cnt_dict = {k:(qm9_gibbs_order[k-1][0], qm9_gibbs_order[k-1][1], qm9_gibbs_order[k-1][2]) for k,v in sorted(id_cnt_dict.items(), key=lambda item: item[1])}
        
        order_list = []
        for prdct_idx, (k, v) in enumerate(id_cnt_dict.items()):
            order_list.append((prdct_idx, v[1]))
        print('order list (predict, actual): ' + str(order_list))
        predict_actual_order_list += order_list

        meg_order_list = []
        for prdct_idx, (k, v) in enumerate(id_megnet_gibbs_dict.items()):
            meg_order_list.append((prdct_idx, qm9_gibbs_order[k-1][1]))
        print('meg order list (predict, actual): ' + str(meg_order_list))
        megnet_predict_actual_order_list += meg_order_list

        print('Actual: id, ranking in Gibbs, totoal cnts: ')
        print(id_cnt_dict)
        print('chemical formula: ', last_formula)
        last_formula = rows[id_with_ret[i][0]-1].formula
        id_cnt_dict = {}
        id_megnet_gibbs_dict = {}
        print('-' * 100 )
    if(rows[id_with_ret[i][0]-1].id in id_cnt_dict.keys()):
        if(id_with_ret[i][2] == 1):
            id_cnt_dict[rows[id_with_ret[i][0]-1].id] += 1
    else:
        id_cnt_dict[rows[id_with_ret[i][0]-1].id] = 0

    id_megnet_gibbs_dict[rows[id_with_ret[i][0]-1].id] = megnet_gibbs[rows[id_with_ret[i][0]-1].id]
    if(i == len(id_with_ret) - 1):
        # sort a dict by value
        id_cnt_dict = {k:v for k,v in sorted(id_cnt_dict.items(), key=lambda item: item[1])}
        print('Prediction: id, total loss cnt: ')
        print(id_cnt_dict)
 
        id_megnet_gibbs_dict = {k:v for k,v in sorted(id_megnet_gibbs_dict.items(), key=lambda item: item[1])}
        print(' megnet Prediction: id, total loss cnt: ')
        print( id_megnet_gibbs_dict)

        id_cnt_dict = {k:(qm9_gibbs_order[k-1][0], qm9_gibbs_order[k-1][1], qm9_gibbs_order[k-1][2]) for k,v in sorted(id_cnt_dict.items(), key=lambda item: item[1])}
        
        order_list = []
        for prdct_idx, (k, v) in enumerate(id_cnt_dict.items()):
            order_list.append((prdct_idx, v[1]))
        print('order list (predict, actual): ' + str(order_list))
        predict_actual_order_list += order_list


        meg_order_list = []
        for prdct_idx, (k, v) in enumerate(id_megnet_gibbs_dict.items()):
            meg_order_list.append((prdct_idx, qm9_gibbs_order[k-1][1]))
        print('meg order list (predict, actual): ' + str(meg_order_list))
        megnet_predict_actual_order_list += meg_order_list


        print('Actual: id, ranking in Gibbs, totoal cnts: ')
        print(id_cnt_dict)
        print('chemical formula: ', last_formula)
        last_formula = rows[id_with_ret[i][0]-1].formula
        id_cnt_dict = {}
        print('-' * 100 )


import pickle
with open('predict_actual_order_lst.txt', 'wb') as fp:
    pickle.dump(predict_actual_order_list, fp)
import pickle
with open('megnet_predict_actual_order_lst.txt', 'wb') as fp:
    pickle.dump(megnet_predict_actual_order_list, fp)
