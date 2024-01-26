from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as func

class NoiseLayer(nn.Module):
    def __init__(self, SNR, alpha, dc=0 , bitdepth=8) -> None:
        super(NoiseLayer,self).__init__()
        self.SNR = SNR
        self.alpha = alpha
        self.dc = dc
        self.bitdepth = bitdepth

    def forward(self, input):
        # input: [batch_size, channel, height, width]

        # calculate the power of the input signal

        input_shape = tuple(input.shape)

        input_power = torch.mean(torch.pow(input, 2))

        sigma = input_power * (10 ** (-self.SNR / 10))

        sigma_sd = sigma * self.alpha / (1+self.alpha)
        sigma_si = sigma / (1+self.alpha)

        RandomVariable_sd = torch.normal(mean=0., std=torch.sqrt(sigma_sd).item(), size=input_shape, device=input.device)

        dependent_noise = torch.sqrt(input) * RandomVariable_sd

        independent_noise = torch.normal(mean=0., std=torch.sqrt(sigma_si).item(), size=input_shape, device=input.device)

        noise = dependent_noise + independent_noise
        noisy_data = input + noise

        # dark current
        if self.dc > 0:
            noisy_data = noisy_data + self.dc

        # quantization
        L = 2 ** self.bitdepth - 1
        noisy_data = torch.round(noisy_data * L) / L
        noisy_data = torch.clamp(noisy_data, min=0)

        return noisy_data


class ClearNoiseLayer(nn.Module):
    def __init__(self, bitdepth=8) -> None:
        super(ClearNoiseLayer,self).__init__()
        self.bitdepth = bitdepth

    def forward(self, input):
        # input: [batch_size, channel, height, width]

        # quantization
        L = 2 ** self.bitdepth - 1
        noisy_data = torch.round(input * L) / L
        noisy_data = torch.clamp(noisy_data, 0, 1)

        return noisy_data