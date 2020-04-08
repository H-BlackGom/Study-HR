import torch
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h


def function_1(x):
    return 0.01 * x **2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))
print("---------------------------------------------------")

def function_2(x):
    return x[0] ** 2 + x[1] ** 2


# center numerical diff
def numerical_gradient(f, x):
    h = 1e-4
    grad = torch.zeros_like(x)

    for idx in range(x.size()[0]):
        tmp_val = x[idx].item()

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad
#
#
# print(numerical_gradient(function_2, torch.tensor([3.0, 4.0])))
# print(numerical_gradient(function_2, torch.tensor([0.0, 2.0])))
# print(numerical_gradient(function_2, torch.tensor([3.0, 0.0])))
# print("---------------------------------------------------")

def numpy_numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


print(numpy_numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numpy_numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numpy_numerical_gradient(function_2, np.array([3.0, 0.0])))

# gradient descent
def numpy_gradient_descent(f, init_x, lr=0.001, step_num=100):
    x = init_x

    for i in range(step_num):
        # grad = numerical_gradient(f, x)
        grad = numpy_numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
print(numpy_gradient_descent(function_2, init_x, lr=10.0, step_num=100))


def gradient_descent(f, init_x, lr=0.001, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = torch.tensor([-3.0, 4.0], dtype=torch.float64)
print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))
print("---------------------------------------------------")

print(gradient_descent(function_2, init_x, lr=10.0, step_num=100))
print(gradient_descent(function_2, init_x, lr=1e-10, step_num=100))
print("---------------------------------------------------")