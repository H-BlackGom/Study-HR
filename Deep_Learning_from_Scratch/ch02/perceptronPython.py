import torch

def AND(x1, x2):
    # Perceptron
    # 0 <- w1*x1 + w2*x2 <= theta
    # 1 <- w1*x1 + w2*x2 > theta
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp > theta:
        return 1
    elif tmp <= theta:
        return 0


print("Perceptron AND")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print("_____________")


# Apply weight and bias
def new_AND(x1, x2):
    # weight and bias perceptron
    # 0 <- w1*x1 + w2*x2 + b <= 0
    # 1 <- w1*x1 + w2*x2 + b > 0

    x = torch.tensor([x1, x2])
    w = torch.tensor([0.5, 0.5])
    b = -0.7

    tmp = torch.sum(x*w) + b
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0


print("Perceptron AND with weight and bias")
print(new_AND(0, 0))
print(new_AND(1, 0))
print(new_AND(0, 1))
print(new_AND(1, 1))
print("_____________")


def OR(x1, x2):
    x = torch.tensor([x1, x2])
    w = torch.tensor([0.5, 0.5])
    b = -0.2

    tmp = torch.sum(x * w) + b
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0


print("Perceptron OR with weight and bias")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
print("_____________")


def NAND(x1, x2):
    x = torch.tensor([x1, x2])
    w = torch.tensor([-0.5, -0.5])
    b = 0.7

    tmp = torch.sum(x * w) + b
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0


print("Perceptron NAND with weight and bias")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
print("_____________")


# Multi-layer perceptron
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("Multi-layer perceptron XOR with weight and bias")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
print("_____________")