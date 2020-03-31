import sys, os
import torch
import numpy as np
import pickle
from PIL import Image
from Deep_Learning_from_Scratch.ch03.NeuralNetMNIST_Sample.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
x_train = torch.from_numpy(x_train)
t_train = torch.from_numpy(t_train)
x_test = torch.from_numpy(x_test)
t_test = torch.from_numpy(t_test)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


def img_show(img):
    # TODO: PIL의 Image.fromarray는 torch.tensor array를 인식 하지 못하는가?
    pil_img = Image.fromarray(np.uint8(img.numpy()))
    pil_img.show()


img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
print("----------------------")

