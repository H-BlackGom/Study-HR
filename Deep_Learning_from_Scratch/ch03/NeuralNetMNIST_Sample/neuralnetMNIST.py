import torch
import pickle
import numpy as np
import torch.nn as nn
from Deep_Learning_from_Scratch.ch03.activityFunctions import fn_sigmoid
from Deep_Learning_from_Scratch.ch03.outputFunctions import fn_softMax
from Deep_Learning_from_Scratch.ch03.NeuralNetMNIST_Sample.dataset.mnist import load_mnist


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = {}
        self.init_network()

        self.network['W1'] = torch.from_numpy(self.network['W1'])
        self.network['W2'] = torch.from_numpy(self.network['W2'])
        self.network['W3'] = torch.from_numpy(self.network['W3'])

        self.network['b1'] = torch.from_numpy(self.network['b1'])
        self.network['b2'] = torch.from_numpy(self.network['b2'])
        self.network['b3'] = torch.from_numpy(self.network['b3'])


    def init_network(self):
        with open("sample_weight.pkl", 'rb') as f:
            self.network = pickle.load(f)

    def forward(self, x):
        a1 = x.float()@self.network['W1'] + self.network['b1']
        z1 = fn_sigmoid(a1)
        a2 = z1@self.network['W2'] + self.network['b2']
        z2 = fn_sigmoid(a2)
        a3 = z2@self.network['W3'] + self.network['b3']

        y = fn_softMax(a3)

        return y


def get_data():
    (x_train, t_train), (x_test, t_test) \
        = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    x_train = torch.from_numpy(x_train)
    t_train = torch.from_numpy(t_train)
    x_test = torch.from_numpy(x_test)
    t_test = torch.from_numpy(t_test)

    return x_test, t_test


mnist = NeuralNet()
x, t = get_data()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    y = mnist(x[i: i+batch_size])
    p = torch.argmax(y, axis=1)
    # if p == t[i]:
    #     accuracy_cnt += 1

    accuracy_cnt += torch.sum((p == t[i:i+batch_size]).int())

print("Accuracy: ", accuracy_cnt.float() / len(x))