from ase.db import connect
from ase.io import read, write
from ase.visualize import view
import random
import pymatgen.io.ase as pymatgen_io_ase
import os
import numpy as np

from megnet.models import MEGNetModel
import numpy as np
from operator import itemgetter
import json

from base64 import b64encode, b64decode
#/home/inode01/xiaotong/code/megnet/mvl_models/qm9-2018.6.1 

from megnet.utils.molecule import get_pmg_mol_from_smiles

seed = 1234
random.seed(seed)
np.random.seed(seed)
filename = 'qm9.db'

db = connect(filename)
rows = list(db.select(sort='id'))
structure_dataset = []
MODEL_NAME = 'G'

G = "free_energy"
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

model = MEGNetModel.from_file('/home/inode01/xiaotong/code/megnet/mvl_models/qm9-2018.6.1/%s.hdf5' % MODEL_NAME)
import math

atom_energy = {'H': -0.510927, 'C': -37.861317, 'N': -54.598897, 'O': -75.079532, 'F': -99.733544}
megnet_lst = []

print('id', 'energy megnet', 'energy in db')
for row in rows:
    #  print(row.id)
    #  if(row.id > 130):
    #      break
    atoms = row.toatoms()
    atoms.set_cell(100 * np.identity(3)) # if don't set_cell, later converter will crash..
    stru = pymatgen_io_ase.AseAtomsAdaptor.get_structure(atoms)
    pred_target = model.predict_structure(stru)
    # pred_target = list(pred_target)
    if(row.id%1000==999 or math.isnan(pred_target[0])):
        molecule_gibbs = get_data_pp(row.id, G)[0]
        for at in atoms:
            molecule_gibbs -= (atom_energy[at.symbol] * 27.2116)

        print(row.id, pred_target, molecule_gibbs)
        print('-'*100)
    megnet_lst.append((row.id, pred_target[0]))

import pickle
with open('megnet_gibbs.lst', 'wb') as fp:
    pickle.dump(megnet_lst, fp)
# The smiles of qm9:000001 is just C
# mol1 = get_pmg_mol_from_smiles('C')
# ret = model.predict_structure(mol1)
# print(ret)
# print('finish..')

