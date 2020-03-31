import torch
import torch.nn as nn
from Deep_Learning_from_Scratch.ch03.activityFunctions import fn_sigmoid


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = {}
        self.network['W1'] = torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.network['b1'] = torch.tensor([0.1, 0.2, 0.3])
        self.network['W2'] = torch.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.network['b2'] = torch.tensor([0.1, 0.2])
        self.network['W3'] = torch.tensor([[0.1, 0.3], [0.2, 0.4]])
        self.network['b3'] = torch.tensor([0.1, 0.2])

    def forward(self, x):
        a = x@self.network['W1'] + self.network['b1']
        z = fn_sigmoid(a)
        a1 = z@self.network['W2'] + self.network['b2']
        z1 = fn_sigmoid(a1)
        a2 = z1@self.network['W3'] + self.network['b3']
        y = self.fn_identity(a2)

        return y

    def fn_identity(self, x):
        return x


net = NeuralNet()
x = torch.tensor([1.0, 0.5])
y = net(x)
print(y)
