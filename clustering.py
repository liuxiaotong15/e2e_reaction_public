import math
import numpy as np
from progress.bar import Bar
from ase import Atoms
from itertools import combinations
from ase.db import connect
from ase.visualize import view
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import numpy as np
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

import pickle
with open('data_dict.lst', 'rb') as fp:
    data_dict = pickle.load(fp)

clst_dict = {}
total_clst_cnt = 0
# import numpy as np
# from sklearn.datasets.samples_generator import make_blobs
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)
# X, y = make_blobs(n_samples=1000, n_features=3, centers=[[-1,-1,-1], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)
# X, y = make_blobs(n_samples=1000, n_features=1, centers=[[-1], [0], [1], [2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)

for k in data_dict.keys():
    patience = 0
    X = data_dict[k]
    if len(X) < 1000:
        continue
    print(k, len(X))
    # mean-shift
    label_unique = []
    # bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=10000, random_state=9)
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms.fit(X)
    # labels = ms.labels_
    # label_unique = np.unique(labels)
    # if(len(label_unique) > 1):
    #     print('MS: ', k, len(label_unique), metrics.silhouette_score(X, labels, sample_size=10000), len(X))
    # total_clst_cnt += len(label_unique)
    # clst_dict[k] = ms.cluster_centers_
    # k-mean
    best_score = 0
    best_clst_cnts = 2
    for n_clst in range(2, 100):
        if n_clst > 5 and n_clst % 2 == 0:
            continue 
        km = KMeans(n_clusters=n_clst, random_state=9)
        y_pred = km.fit_predict(X)
        # print(metrics.calinski_harabaz_score(X, y_pred))
        slht = metrics.silhouette_score(X, y_pred, sample_size=1000)
        # slht = metrics.calinski_harabasz_score(X, y_pred)
        if(slht > best_score):
            best_score = slht
            best_clst_cnts = n_clst
            patience = 0
        else:
            patience += 1
        if patience > 5 and n_clst>10:
            break
    print('KM: ', k, best_clst_cnts, best_score, len(X))
    km = KMeans(n_clusters=best_clst_cnts, random_state=9)
    y_pred = km.fit_predict(X)
    total_clst_cnt += max(best_clst_cnts, len(label_unique))
    if best_clst_cnts > len(label_unique):
        clst_dict[k] = km.cluster_centers_

print(total_clst_cnt)

with open('clst_dict.dct', 'wb') as fp:
    pickle.dump(clst_dict, fp)
