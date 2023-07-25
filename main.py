import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data_manager as dm

# data
def next_data(*, sample_n, harmonic_n, channel_n, dev):
    target = dm.get_clean(length=sample_n, max_harmonics=harmonic_n, channels=channel_n)
    data_in = dm.add_noise(samples=target, max_amplitude=0.1)
    return data_in.to(dev), target.to(dev)


train = True

# device
dev = torch.device("cpu")

if torch.cuda.is_available():
    dev = torch.device("cuda")

if torch.backends.mps.is_available():
    dev = torch.device("mps")

if train:
    # model
    model = nn.Sequential(
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=128)
    ).to(dev)

    print(f"Device: {dev}")
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    epoch_n = 128

    loss_sum = 0

    for epoch in range(1, epoch_n + 1):
        data_in, target = next_data(sample_n=128, harmonic_n=16, channel_n=128, dev=dev)
        prediction = model.forward(data_in)

        optimizer.zero_grad()
        loss = criterion(target, prediction)
        loss.backward()
        optimizer.step()
        print(float(loss))

    torch.save(obj=model, f="model.pt")


else:
    # model
    model = torch.load(f="model.pt").to(dev)

    print(f"Device: {dev}")
    print(model)

    while 2137:
        data_in, target = next_data(1, dev)
        prediction = model.forward(data_in)
        plt.plot(data_in[0].cpu().detach().numpy(), "r")
        plt.plot(target[0].cpu().detach().numpy(), "g")
        plt.plot(prediction[0].cpu().detach().numpy(), "b")
        plt.show()

