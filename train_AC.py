import pickle
import numpy as np

import random

seed = 1234
random.seed(seed)
np.random.seed(seed)


with open('dataset.lst', 'rb') as fp:
    dataset = pickle.load(fp)
with open('test_dataset.lst', 'rb') as fp:
    test_dataset = pickle.load(fp)

print('input BoC length is: ', len(dataset[0][0]))

random.shuffle(dataset)
zero_cnt = 0
data = []
label = []
for i in range(len(dataset)):
    if(dataset[i][1]<0):
        dataset[i][1] = 0
        zero_cnt += 1
    data.append(dataset[i][0])
    label.append(dataset[i][1])
print(zero_cnt)
data = np.array(data, dtype=np.float32)
label = np.array(label)
# change -1 to 0, using softmax

competition_data = []
for i in range(len(test_dataset)):
    competition_data.append(test_dataset[i][2])
competition_data = np.array(competition_data, dtype=np.float32)

import torch

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(dataset[0][0]), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)
batch_size = 1024
 
train_sets = data[:int(len(data)*0.8)]
vali_sets = data[int(len(data)*0.8): int(len(data)*0.9)]
test_sets = data[int(len(data)*0.9):]
train_labels = label[:int(len(data)*0.8)]
vali_labels = label[int(len(data)*0.8): int(len(data)*0.9)]
test_labels = label[int(len(data)*0.9):]

min_vali_loss = 9999
patience = 10
cur_pat = patience

for epoch in range(2000):  # loop over the dataset multiple times
    if(cur_pat == 0):
        break
    running_loss = 0.0
    # train
    for i in range(0, len(train_labels), batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.from_numpy(train_sets[i:i+batch_size]).to('cpu')
        # inputs = torch.from_numpy(train_sets[i:min(i+batch_size, len(label))]).to('cpu')
        # labels = torch.from_numpy(train_labels[i:min(i+batch_size, len(label))]).to('cpu')
        labels = torch.from_numpy(train_labels[i:i+batch_size]).to('cpu')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 20 == 19:    # print every 2000 mini-batches
        if((i/batch_size)%1000 == 999):
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss/1000))
            running_loss = 0.0
    # valid
    with torch.no_grad():
        N = len(vali_sets)
        SAE = 0
        success = 0
        failure = 0
        for i in range(0, N, batch_size):
            inputs = torch.from_numpy(vali_sets[i:i+batch_size]).to('cpu')
            labels = torch.from_numpy(vali_labels[i:i+batch_size]).to('cpu')

            vali_outputs = net(inputs)
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
        MAE = SAE/N
        if(MAE < min_vali_loss):
            min_vali_loss = MAE
            cur_pat = patience
        else:
            cur_pat -= 1
    print('vali acc: ' + str(success/(success+failure) * 100) + '%, vali loss: ' + str(MAE))
    # test
    with torch.no_grad():
        N = len(test_sets)
        SAE = 0
        l1p1 = 0
        l1p0 = 0
        l0p1 = 0
        l0p0 = 0
        success = 0
        failure = 0
        for i in range(0, N, batch_size):
            inputs = torch.from_numpy(test_sets[i:i+batch_size]).to('cpu')
            labels = torch.from_numpy(test_labels[i:i+batch_size]).to('cpu')

            test_outputs = net(inputs)
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
        MAE = SAE/N
    print('test acc: ' + str(success/(success+failure) * 100) + '%, test loss: ' + str(MAE))
    print('l1p1: ' , l1p1 , ' l1p0: ' , l1p0 , ' l0p1: ' , l0p1 , 'l0p0: ' , l0p0)

    cwr_lst = []
    # competition start here
    with torch.no_grad():
        N = len(competition_data)
        total_p = []
        for i in range(0, N, batch_size):
            inputs = torch.from_numpy(competition_data[i:i+batch_size]).to('cpu')
            # labels = torch.from_numpy(test_labels[i:i+batch_size]).to('cpu')

            competition_outputs = net(inputs)
            _, predict_idx = torch.max(competition_outputs, 1)
            p = predict_idx.tolist()
            total_p += p
        print('output label length is: ', len(total_p))
        print('competition_data length is: ', N)
        print('id1, id2, deltaBoC count:' ,len(test_dataset))
        for i in range(len(test_dataset)):
            cwr_lst_item = []
            cwr_lst_item.append(test_dataset[i][0])
            cwr_lst_item.append(test_dataset[i][1])
            cwr_lst_item.append(total_p[i])
            cwr_lst.append(cwr_lst_item)
            # test_dataset[i].append(total_p[i])

    with open('competition_with_ret_'+str(epoch)+'.txt', 'wb') as fp:
        pickle.dump(cwr_lst, fp)

print('Finished Training')
