import data_manager
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

samples_n = 128
model = torch.load("model.pth")
harmonics = 8
conv_a = [0.1] * 10

while 2137:
    target = data_manager.get_clean(samples_n, harmonics)
    data_in = data_manager.add_noise(target, 0.1)
    result = model.forward(data_in)
    conv_res = np.convolve(data_in, conv_a, mode = "same")

    plt.plot(target, color = "blue", linewidth = 2)
    plt.plot(data_in, color = "red", linewidth = 1)
    plt.plot(result.detach().numpy(), color = "green", linewidth = 4)
    plt.show()

    plt.plot(target, color = "blue", linewidth = 2)
    plt.plot(data_in, color = "red", linewidth = 1)
    plt.plot(conv_res, color = "green", linewidth = 4)
    plt.show()
