import torch


def fn_identity(a):
    return a


def fn_softMax(a):
    c = torch.max(a)
    return torch.exp(a-c) / torch.sum(torch.exp(a-c))


a = torch.tensor([0.3, 2.9, 4.0])
y = fn_softMax(a)
print(y)
print(torch.sum(y))
