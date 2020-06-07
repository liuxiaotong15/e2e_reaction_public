import pickle
import h5py
import numpy as np
import os
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

cur_file_id = 1
data_pool_size = 2e6

file_cnt = 30
step_cnt = 1e6

data_arr = None
label_arr = None
BoC_size = 0

data_pool = None
label_pool = None
train_cnt = int(8e7)
vali_cnt = int(1e7)
test_cnt = int(1e7)

def load_data_fill_pool():
    global cur_file_id, BoC_size, data_pool, label_pool
    f = h5py.File('dataset_new_' + str(cur_file_id) + '.hdf5', 'r')
    # print('start to load data.')
    dataset = f['dset1'][:]
    # print('start to shuffle data.')
    # np.random.shuffle(dataset) # the data distribution is not uniform last 10% maybe all 1, so must shuffle here
    zero_cnt = 0
    data = []
    label = []
    for j in range(len(dataset)):
        if(dataset[j][-1]<0):
            # change -1 to 0, using softmax
            dataset[j][-1] = 0
            zero_cnt += 1
        data.append(dataset[j][0:-1])
        label.append(int(dataset[j][-1]))
    # print(zero_cnt, 'start to cat data')
    if cur_file_id == 1:
        print('input BoC length is: ', len(dataset[0])-1)
        BoC_size = len(dataset[0])-1
        data_pool = np.array(data, dtype=np.float32)
        label_pool = np.array(label)
    else:
        # we have to concat many times due to the memory size reason... although waste so much time
        data_pool = np.concatenate((data_pool, np.array(data, dtype=np.float32)), axis=0)
        label_pool = np.concatenate((label_pool, np.array(label)), axis=0)
    # data_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(data, dtype=np.float32)
    # label_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(label)
   
    # print(cur_file_id, data_pool.shape, label_pool.shape)
    cur_file_id += 1
    f.close()

# print('Data load finished.')

load_data_fill_pool()

import torch

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(BoC_size, 4096)
        self.dr1 = nn.Dropout(0.5)
        self.fc11 = nn.Linear(4096, 2048)
        self.fc12 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dr2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 128)
        self.dr3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)
        self.dr4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        # x = self.dr1(x)
        x = F.relu(self.fc2(x))
        # x = self.dr2(x)
        x = F.relu(self.fc3(x))
        # x = self.dr3(x)
        x = F.relu(self.fc4(x))
        # x = self.dr4(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

net = Net()
# model_dump_name = '517b931_epoch_11_test_acc_98.75_vali_loss_7.728.model'
# model_dump_name = '3cf4397_epoch_8_test_acc_98.77_vali_loss_7.392.model'
# model_dump_name = 'c5b4037_epoch_0_test_acc_98.99_vali_loss_6.765.model'
# net.load_state_dict(torch.load(model_dump_name))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
lr = 0.001
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
# batch_size = 1024
batch_size = 4096
 
# train_sets = data_arr[:int(len(data_arr)*0.8)]
# vali_sets = data_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)]
# test_sets = data_arr[int(len(data_arr)*0.9):]
# train_labels = label_arr[:int(len(data_arr)*0.8)]
# vali_labels = label_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)]
# test_labels = label_arr[int(len(data_arr)*0.9):]

min_vali_loss = 9999
patience = 10
cur_pat = patience
commit_id = str(os.popen('git --no-pager log -1 --oneline').read()).split(' ', 1)[0]
for epoch in range(2000):  # loop over the dataset multiple times
    if epoch > 0:
        cur_file_id = 1
    running_loss = 0.0
    data_consumed = 0
    # train
    while(data_consumed < train_cnt):
        success = 0
        failure = 0
        net.train()
    # for i in range(0, len(train_labels), batch_size):
        # get the inputs; data is a list of [inputs, labels]
        while(label_pool.shape[0] < min(batch_size, train_cnt-data_consumed)):
            load_data_fill_pool()
        inputs = torch.from_numpy(data_pool[0:min(batch_size, train_cnt-data_consumed)]).to('cpu')
        # inputs = torch.from_numpy(train_sets[i:min(i+batch_size, len(label))]).to('cpu')
        # labels = torch.from_numpy(train_labels[i:min(i+batch_size, len(label))]).to('cpu')
        labels = torch.from_numpy(label_pool[0:min(batch_size, train_cnt-data_consumed)]).to('cpu')
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(inputs.float())
        _, predict_idx = torch.max(outputs, 1)
        p = predict_idx.tolist()
        c = labels.tolist()
        # print(correct_properties)
        for j in range(len(p)):
            if(p[j] == c[j]):
                success+=1
            else:
                failure+=1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 20 == 19:    # print every 2000 mini-batches
        if((data_consumed/batch_size)%int(1e3) == int(1e3-1)):
            print(commit_id + ' train: [%d, %5d] loss: %.6f, acc: %.6f' %
                  (epoch + 1, data_consumed + 1, running_loss/int(1e3), success/(success+failure)))
            running_loss = 0.0
        
        step_len = min(batch_size, train_cnt-data_consumed)
        data_pool = data_pool[step_len:]
        label_pool = label_pool[step_len:]
        data_consumed += step_len

    # valid
    with torch.no_grad():
        net.eval()
        # N = len(vali_sets)
        N = vali_cnt
        SAE = 0
        success = 0
        failure = 0
        # for i in range(0, N, batch_size):
        while(data_consumed < train_cnt + vali_cnt):
            while(label_pool.shape[0] < min(batch_size, train_cnt + vali_cnt -data_consumed)):
                load_data_fill_pool()

            inputs = torch.from_numpy(data_pool[0:min(batch_size, train_cnt + vali_cnt -data_consumed)]).to('cpu')
            labels = torch.from_numpy(label_pool[0:min(batch_size, train_cnt + vali_cnt -data_consumed)]).to('cpu')

            vali_outputs = net(inputs.float())
            # print(vali_outputs.shape, labels.shape)
            loss = criterion(vali_outputs, labels)
            _, predict_idx = torch.max(vali_outputs, 1)
            p = predict_idx.tolist()
            c = labels.tolist()
            # print(correct_properties)
            for j in range(len(p)):
                if(p[j] == c[j]):
                    success+=1
                else:
                    failure+=1
            SAE += loss.item()
            step_len = min(batch_size, train_cnt + vali_cnt-data_consumed)
            data_pool = data_pool[step_len:]
            label_pool = label_pool[step_len:]
            data_consumed += step_len
        MAE = SAE/N
        if(MAE < min_vali_loss):
            min_vali_loss = MAE
            cur_pat = patience
        else:
            cur_pat -= 1
    vali_loss=str(MAE)[:5]
    print('epoch: ' + str(epoch+1) + ', vali acc: ' + str(success/(success+failure) * 100) + '%, vali loss: ' + str(MAE))
    # test
    with torch.no_grad():
        net.eval()
        N = test_cnt
        SAE = 0
        l1p1 = 0
        l1p0 = 0
        l0p1 = 0
        l0p0 = 0
        success = 0
        failure = 0
        while(data_consumed < train_cnt + vali_cnt + test_cnt):
            while(label_pool.shape[0] < min(batch_size, train_cnt + vali_cnt + test_cnt -data_consumed)):
                load_data_fill_pool()

            inputs = torch.from_numpy(data_pool[0:min(batch_size, train_cnt + vali_cnt + test_cnt -data_consumed)]).to('cpu')
            labels = torch.from_numpy(label_pool[0:min(batch_size, train_cnt + vali_cnt + test_cnt -data_consumed)]).to('cpu')

            test_outputs = net(inputs.float())
            loss = criterion(test_outputs, labels)
            _, predict_idx = torch.max(test_outputs, 1)
            p = predict_idx.tolist()
            l = labels.tolist()
            # print(correct_properties)
            for j in range(len(p)):
                if(p[j] == l[j]):
                    success+=1
                else:
                    failure+=1
                if(l[j] == 1 and p[j] == 1):
                    l1p1 += 1
                elif(l[j] == 1 and p[j] == 0):
                    l1p0 += 1
                elif(l[j] == 0 and p[j] == 1):
                    l0p1 += 1
                elif(l[j] == 0 and p[j] == 0):
                    l0p0 += 1
                else:
                    pass
            SAE += loss.item()
            step_len = min(batch_size, train_cnt + vali_cnt + test_cnt-data_consumed)
            data_pool = data_pool[step_len:]
            label_pool = label_pool[step_len:]
            data_consumed += step_len

        MAE = SAE/N
    test_acc = str(success/(success+failure) * 100)[:5]
    print('epoch: ' + str(epoch+1) + ', test acc: ' + str(success/(success+failure) * 100) + '%, test loss: ' + str(MAE))
    print('l1p1: ' , l1p1 , ' l1p0: ' , l1p0 , ' l0p1: ' , l0p1 , 'l0p0: ' , l0p0)
    # if(cur_pat == patience):
    # save model with name: commit id
    model_dump_name = commit_id + '_epoch_' + str(epoch) + '_test_acc_' + test_acc + '_vali_loss_' + vali_loss + '.model'
    torch.save(net.state_dict(), model_dump_name)
    if(cur_pat == patience//5):
        # lr = 0.001 * (0.1 ** (epoch//15))
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if(cur_pat == 0):
        break
print('Finished Training')
