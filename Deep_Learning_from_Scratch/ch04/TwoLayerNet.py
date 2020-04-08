import torch
import numpy as np


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softMax(a):
    c = torch.max(a)
    return torch.exp(a-c) / torch.sum(torch.exp(a-c))


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


def cross_entropy_error(y, t):
    if y.dim() == 1:
        y = y.view(1, y.size()[0])
        t = t.view(1, t.size()[0])

    if t.size() == y.size():
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(batch_size), t] + 1e-7)) / batch_size


# ToDo: torch module로 변경
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * torch.randn(input_size, hidden_size)
        self.params['b1'] = torch.zeros(hidden_size)
        self.params['W2'] = weight_init_std * torch.randn(hidden_size, output_size)
        self.params['b2'] = torch.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = x@W1 + b1
        z1 = sigmoid(a1)
        a2 = z1@W2 + b2
        y = softMax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = torch.argmax(y, axis=1)
        t = torch.argmax(t, axis=1)

        accuracy = (torch.sum(y == t) / x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads