import torch
import matplotlib.pyplot as plt

def get_clean(length, max_harmonics):
    harmonics = torch.randint(1, max_harmonics, [1])
    output = torch.tensor([0.0] * length)
    t = torch.linspace(0, 1, length)
    for i in range(harmonics):
        output += torch.rand(1) * torch.sin(2 * torch.pi * harmonics * torch.rand(1) * (t + torch.rand(1)))

    output -= torch.min(output)
    output /= torch.max(output)
    p1 = 1.8 * torch.rand(1) - 0.9
    p2 = 1.8 * torch.rand(1) - 0.9

    output *= (p1 - p2)
    output += p2
    return output


def add_noise(samples, max_amplitude):
    return samples + max_amplitude * torch.rand(1) * torch.randn(len(samples))

'''
samples_n = 128
harmonics = 4
noise_level = 0.1

while 2137:
    target = get_clean(samples_n, harmonics)
    data_in = add_noise(target, noise_level)

    plt.plot(target, color = "blue", linewidth = 2)
    plt.plot(data_in, color = "red", linewidth = 1)
    plt.show()
'''