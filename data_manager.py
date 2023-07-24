import torch
import matplotlib.pyplot as plt

def get_clean(*, length, max_harmonics, channels):
    harmonics = torch.randint(1, max_harmonics, [1])
    output = torch.zeros((channels, length))
    t = torch.linspace(0, 1, length)
    for i in range(harmonics):
        output += torch.sin(2 * torch.pi * harmonics * torch.rand((channels, 1)) * (t + torch.rand(1)))

    output -= torch.min(output)
    output /= torch.max(output)
    p1 = 1.8 * torch.rand((channels, 1)) - 0.9
    p2 = 1.8 * torch.rand((channels, 1)) - 0.9

    output *= (p1 - p2)
    output += p2
    return output


def add_noise(*, samples, max_amplitude, channels):
    return samples + max_amplitude * (2 * torch.rand((channels, len(samples[0]))) - 1)


samples_n = 128
harmonics = 16
noise_level = 0.1

while 2137:
    target = get_clean(length=samples_n, max_harmonics=harmonics, channels=4)
    data_in = add_noise(samples=target, max_amplitude=noise_level, channels=4)

    plt.plot(target[0], color = "blue", linewidth = 2)
    plt.plot(data_in[0], color = "red", linewidth = 1)
    plt.show()
