import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data_manager as dm

# data
def next_data(channels, dev):
    target = dm.get_clean(length=128, max_harmonics=16, channels=channels)
    data_in = dm.add_noise(samples=target, max_amplitude=0.1)
    return data_in.to(dev), target.to(dev)


train = True

if train:
    # model
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

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

    epoch_n = 1024

    loss_sum = 0

    for epoch in range(1, epoch_n + 1):
        data_in, target = next_data(1024, dev)
        prediction = model.forward(data_in)

        optimizer.zero_grad()
        loss = criterion(target, prediction)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss)
        print(loss_sum)
        print(epoch)

    torch.save(obj=model, f="model.pt")


else:
    # model
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

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

