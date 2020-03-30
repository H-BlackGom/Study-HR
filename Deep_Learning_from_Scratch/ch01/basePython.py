###########################################
# 기초적인 python code 사용법은 생략.
# numpy 대신 Pytorch의 tensor로 예제를 진행.
# matplotlib 사용법 생략. -> 그래프는 되도록 tensorboard를 사용 예정.
#
# 2020.03.30
# h.blackgom@gmail.com
###########################################
# pytorch 가져오기
import torch

# Create tensor array
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

print(x, type(x))
print(y, type(y))
print("_____________")

# Operation tensor (Vector)
print(x + y, "(Vector) Sum of tensor")
print(x - y, "(Vector) Sub of tensor")
print(x * y, "(Vector) Mul of tensor")
print(x / y, "(Vector) Division of tensor")
print("_____________")

# Operation tensor (Scalar)
# broadcast
print(x + 2, "(Scalar) Sum of tensor")
print(x - 1, "(Scalar) Sub of tensor")
print(x * 2, "(Scalar) Mul of tensor")
print(x / 2, "(Scalar) Division of tensor")
print("_____________")

# Create two dimension tensor (Vector)
x = torch.tensor([[1, 2], [3, 4]])
print(x, type(x))
# Check shape
print(x.shape)
print("_____________")

# N dimension tensor operation (Vector)
y = torch.tensor([[3, 1], [1, 6]])
print(x + y, "(Vector) Sum of tensor")
print(x - y, "(Vector) Sub of tensor")
print(x * y, "(Vector) Mul of tensor")
print(x / y, "(Vector) Division of tensor")
print("_____________")

# Access N dimension according to index
X = torch.tensor([[51, 55], [14, 18], [8, 4]])
print(X, "X vector")
print(X[0], "X[0] elements")
print(X[0][1], "X[0][1] element")

for i, ele in enumerate(X):
   print(ele, "get X[{0}] elements according to for loop".format(i))
print("_____________")

# Dimension flatten
X = X.flatten()
print(X)

# Dimension boolean condition
print(X > 15)
print(X[X > 15])


