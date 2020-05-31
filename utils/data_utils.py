import torch
import numpy as np


def wavenet_max_fieldofvision(max_dilation, n_layers, kernel_size):
    dilations = []
    for i in range(n_layers):
        dilations.append(2 ** (i % max_dilation))

    field = 1 + kernel_size - 1
    for dilation in reversed(dilations):
        field += (kernel_size - 1) * dilation

    return field

def mu_law_decode(x, mu_quantization=256):
    assert(torch.max(x) <= mu_quantization)
    assert(torch.min(x) >= 0)
    x = x.float()
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**torch.abs(signal) - 1)
    return torch.sign(signal) * magnitude

def mu_law_encode(x, mu_quantization=256):
    assert(torch.max(x) <= 1.0)
    assert(torch.min(x) >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).long()
    return encoding.cpu().numpy()

def probability_distribution_expectation(prob_dist, weight, min_prob):
  assert (prob_dist.size() == weight.size())

  prob_recept = prob_dist > min_prob
  prob_recept = prob_recept.float()

  mean = torch.sum(prob_recept * prob_dist * weight) / torch.sum(prob_recept * prob_dist)
  return mean












