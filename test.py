import data_manager
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

samples_n = 128
model = torch.load("model.pth")
harmonics = 8
filter_len = 12
conv_a = [1/filter_len] * filter_len
noise_level = 0.5

loss_nn = 0
loss_filter = 0
criterium = nn.MSELoss()

def MSE(x1, x2):
    x1 = x1.detach().numpy()
    x2 = x2.detach().numpy()
    return np.mean(np.dot(x1 - x2, x1 - x2))


rounds = 10000
for i in range(rounds):
    target = data_manager.get_clean(samples_n, harmonics)
    data_in = data_manager.add_noise(target, noise_level)
    result = model.forward(data_in)
    conv_res = torch.tensor(np.convolve(data_in, conv_a, mode = "same"))
    loss_nn += MSE(target[filter_len:-filter_len], result[filter_len:-filter_len])
    loss_filter += MSE(target[filter_len:-filter_len], conv_res[filter_len:-filter_len])

print("Neural Network MSE:", loss_nn / rounds)
print("Low pass filter MSE:", loss_filter / rounds)

while 2137:
    target = data_manager.get_clean(samples_n, harmonics)
    data_in = data_manager.add_noise(target, noise_level)
    result = model.forward(data_in)[filter_len:-filter_len]
    conv_res = np.convolve(data_in, conv_a, mode = "same")[filter_len:-filter_len]
    target = target[filter_len:-filter_len]
    data_in = data_in[filter_len:-filter_len]

    plt.plot(target, color = "blue", linewidth = 2)
    plt.plot(data_in, color = "red", linewidth = 1)
    plt.plot(result.detach().numpy(), color = "green", linewidth = 4)
    plt.title("Neural Network denosing")
    plt.show()

    plt.plot(target, color = "blue", linewidth = 2)
    plt.plot(data_in, color = "red", linewidth = 1)
    plt.plot(conv_res, color = "green", linewidth = 4)
    plt.title("Low pass filter denosing")
    plt.show()
