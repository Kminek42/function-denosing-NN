import torch

def get_clean(length, max_harmonics):
    harmonics = torch.randint(1, max_harmonics, [1])
    output = torch.tensor([0.0] * length)
    t = torch.linspace(0, 2 * torch.pi, length)
    for i in range(harmonics):
        output += torch.sin(2 * torch.pi * torch.rand(1) + harmonics * torch.rand(1) * t) / (1 + harmonics * harmonics * torch.rand(1))

    return output


def add_noise(samples, max_amplitude):
    return samples + max_amplitude * torch.rand(1) * torch.randn(len(samples))
