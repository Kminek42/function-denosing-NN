import data_manager
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

samples_n = 128
layer1_size = 64
layer2_size = 32
bottleneck_n = 16
activation = nn.LeakyReLU()

# compress: (64, 32)
# loss: 0.004384759347885847
# 0.4663149960516021

model = nn.Sequential(
    nn.Linear(samples_n, layer1_size),
    activation,
    nn.Linear(layer1_size, layer2_size),
    activation,
    nn.Linear(layer2_size, bottleneck_n),
    activation,
    nn.Linear(bottleneck_n, layer2_size),
    activation,
    nn.Linear(layer2_size, layer1_size),
    activation,
    nn.Linear(layer1_size, samples_n)
)
    
rounds = 1000000

criterium = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

d = rounds // 100
loss_sum = 0
y_loss = []

harmonics = 8
noise_level = 0.5

for i in range(rounds):
    target = data_manager.get_clean(samples_n, harmonics)
    data_in = data_manager.add_noise(target, noise_level)
    # data_in = target
    result = model.forward(data_in)

    optimizer.zero_grad()
    loss = criterium(result, target)
    loss.backward()
    optimizer.step()
    loss_sum += loss

    if i % d == 0 and i:
        print(f"progress: {100 * i / rounds}%")
        print(f"loss: {loss_sum / d}")
        y_loss.append(loss_sum.detach().numpy() / d)
        loss_sum = 0

torch.save(model, "model.pth")
plt.plot(y_loss)
plt.show()
