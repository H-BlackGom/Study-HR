import torch
import numpy as np
import matplotlib.pylab as plt


def fn_sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def fn_step(x):
    x = x > 0
    return x.int()


def fn_ReLU(x):
    x = x.numpy()
    # TODO: Torch에도 해당 기능을 하는 function이 있는지 확인.
    y = np.maximum(0, x)
    y = torch.from_numpy(y)
    return y


# x = np.arange(-5.0, 5.0, 0.1)
# x = torch.from_numpy(x)
# y = fn_step(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()
#
# x = np.arange(-5.0, 5.0, 0.1)
# x = torch.from_numpy(x)
# y = fn_sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()
#
# x = np.arange(-5.0, 5.0, 0.1)
# x = torch.from_numpy(x)
# y = fn_ReLU(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()
