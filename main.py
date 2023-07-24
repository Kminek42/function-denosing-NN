import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# data
X = torch.zeros((3, 8))
print(X)
t = torch.linspace(0, 1, 8)
print(t)

X += torch.sin(torch.rand(size=(3, 1)) * (t + torch.rand((3, 1))))

print(X)
print(t)