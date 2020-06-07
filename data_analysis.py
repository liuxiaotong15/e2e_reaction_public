import pickle
import numpy as np

import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

with open('megnet_predict_actual_order_lst.txt', 'rb') as fp:
# with open('predict_actual_order_lst.txt', 'rb') as fp:
    pa_pair = pickle.load(fp)

detail_cnt = 11

cnt = np.zeros((detail_cnt, detail_cnt), dtype=np.float)

for i, j in pa_pair:
    print(i, j)
    i = min(i, detail_cnt-1)
    j = min(j, detail_cnt-1)
    if(i<detail_cnt and j<detail_cnt):
        cnt[detail_cnt -1 - i][j] += 1

# follow GMR's advise, change to % of cols
sum_of_cols = [sum(x) for x in zip(*cnt)]
print(sum_of_cols)

for i in range(detail_cnt):
    for j in range(detail_cnt):
        cnt[i][j] = round(float(cnt[i][j])/sum_of_cols[j]*100, 2)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

ax = plt.subplot(111)
cm = plt.cm.get_cmap('viridis_r')
im = ax.imshow(cnt, cmap=cm, vmin=0, vmax=100)
# im = ax.imshow(cnt)
xticks = range(0, detail_cnt, 1)
xlabels = [el+1 for el in range(detail_cnt)]
xlabels[len(xlabels)-1] = '10+'
ylabels = [detail_cnt - el for el in range(detail_cnt)]
ylabels[0] = '10+'
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_xticklabels(xlabels, font_axis)
ax.set_yticklabels(ylabels, font_axis)
ax.set_ylabel('Prediction Order', font_axis)
ax.set_xlabel('Actual Order', font_axis)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
# Loop over data dimensions and create text annotations.
for i in range(detail_cnt):
    for j in range(detail_cnt):
        if(cnt[i][j] > 1):
            text = ax.text(j, i, int(cnt[i][j]),
                       ha="center", va="center", color="w")
        elif(cnt[i][j] > 0):
            text = ax.text(j, i, cnt[i][j],
                       ha="center", va="center", color="w")


cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Percent in Column, %', fontdict=font_axis)
cbar.ax.tick_params(labelsize=font_axis['size']) 
plt.subplots_adjust(bottom=0.11, right=0.96, left=0.00, top=0.97)
plt.show()
