import torch

# N dimension
A = torch.tensor([1, 2, 3, 4])
print("1 dimension")
print(A)
print(A.dim())
print(A.shape)
print(A.shape[0])
print("-----------------------")

B = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("2 dimension")
print(B)
print(B.dim())
print(B.shape)
print(B.shape[0])
print("-----------------------")

# Matrix Operation (Mul)
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

print("Matrix Mul -1")
print(A@B)
print("-----------------------")

A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[1, 2], [3, 4], [5, 6]])

print("Matrix Mul -2")
print(A@B)
print("-----------------------")

X = torch.tensor([1, 2])
W = torch.tensor([[1, 3, 5], [2, 4, 6]])

print("neural network mul")
print(X@W)
print("-----------------------")
