import numpy as np
import torch
import torch.nn


def sum_of_squares_error(y, t):
    return 0.5 * torch.sum((y-t)**2)


t = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = torch.tensor([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(sum_of_squares_error(y, t), "'2'일 확률이 가장 높은 경우")

y = torch.tensor([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(sum_of_squares_error(y, t), "'7'일 확률이 가장 높은 경우")
print("---------------------------------------------------")


def default_cross_entropy_error(y, t):
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta))


t = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y = torch.tensor([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(default_cross_entropy_error(y, t))


y = torch.tensor([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(default_cross_entropy_error(y, t))
print("---------------------------------------------------")


def numpy_cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


t = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(numpy_cross_entropy_error(y, t))


def torch_cross_entropy_error(y, t):
    if y.dim() == 1:
        y = y.view(1, y.size()[0])
        t = t.view(1, t.size()[0])

    batch_size = y.shape[0]
    return -torch.sum(t * torch.log(y + 1e-7)) / batch_size


t = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y = torch.tensor([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(torch_cross_entropy_error(y, t))


def numpy_cross_entropy_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


t = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(numpy_cross_entropy_error_1(y, t))


def cross_entropy_error_1(y, t):
    if y.dim() == 1:
        y = y.view(1, y.size()[0])
        t = t.view(1, t.size()[0])

    if t.size() == y.size():
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(batch_size), t] + 1e-7)) / batch_size


t = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
y = torch.tensor([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(cross_entropy_error_1(y, t))
print("---------------------------------------------------")
