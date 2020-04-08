import torch
import numpy as np
from Deep_Learning_from_Scratch.ch04.LossFunctions import torch_cross_entropy_error
# from Deep_Learning_from_Scratch.ch04.NumericalDiff import numerical_gradient


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = torch.zeros_like(x).to(dtype=torch.float64)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].item()
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


class simpleNet:
    def __init__(self):
        self.W = torch.randn(2, 3).to(dtype=torch.float64)

    def predict(self, x):
        return x@self.W

    def fn_softMax(self, a):
        c = torch.max(a)
        return torch.exp(a - c) / torch.sum(torch.exp(a - c))

    def loss(self, x, t):
        z = self.predict(x)
        y = self.fn_softMax(z)
        loss = torch_cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

x = torch.tensor([0.6, 0.9], dtype=torch.float64)
p = net.predict(x)
print(p)
print(p.argmax())

t = torch.tensor([0, 0, 1])
print(net.loss(x, t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
