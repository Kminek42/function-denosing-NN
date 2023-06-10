import data_manager
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

samples_n = 128
coder_n = 512
compression_n = 128
activation = nn.Tanh()

model = nn.Sequential(
    nn.Linear(samples_n, coder_n),
    activation,
    nn.Linear(coder_n, compression_n),
    activation,
    nn.Linear(compression_n, coder_n),
    activation,
    nn.Linear(coder_n, samples_n)
)

rounds = 1000000

criterium = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 100 / rounds, momentum = 0.9)

d = rounds / 100
loss_sum = 0
y_loss = []

harmonics = 8
for i in range(rounds):
    target = data_manager.get_clean(samples_n, harmonics)
    data_in = data_manager.add_noise(target, 0.1)
    result = model.forward(data_in)

    optimizer.zero_grad()
    loss = criterium(result, target)
    loss.backward()
    optimizer.step()
    loss_sum += loss

    if i % d == 0 and i:
        print(f"progress: {100 * i / rounds}%")
        print(f"loss: {loss_sum / d}%")
        y_loss.append(loss_sum.detach().numpy() / d)
        loss_sum = 0

torch.save(model, "model.pth")
plt.plot(y_loss)
plt.show()
