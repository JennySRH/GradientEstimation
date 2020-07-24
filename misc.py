import math

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import init
from torch.nn.functional import softplus, softmax


class NonlinearInferenceNet(Module):
    def __init__(self, dim_observed, dim_latent):
        super(NonlinearInferenceNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.3)
        self.enc_fc1 = nn.Linear(dim_observed, dim_latent)
        self.enc_fc2 = nn.Linear(dim_latent, dim_latent)
        self.enc_fc3 = nn.Linear(dim_latent, dim_latent)

    def forward(self, x):
        logits = []
        h = []
        h1 = self.lrelu(self.enc_fc1(2 * x - 1.))
        h2 = self.lrelu(self.enc_fc2(h1))
        logit = self.enc_fc3(h2)
        u = torch.rand_like(logit)
        x = (torch.sigmoid(logit) > u).float()
        logits.append(logit)
        h.append(x)
        return logits, h


class NonlinearGenerativeNet(Module):
    def __init__(self, dim_observed, dim_latent):
        super(NonlinearGenerativeNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.3)
        self.dec_fc1 = nn.Linear(dim_latent, dim_latent)
        self.dec_fc2 = nn.Linear(dim_latent, dim_latent)
        self.dec_fc3 = nn.Linear(dim_latent, dim_observed)

    def forward(self, x):
        logits = []
        h1 = self.lrelu(self.dec_fc1(2 * x[0] - 1.))
        h2 = self.lrelu(self.dec_fc2(h1))
        logit = self.dec_fc3(h2)
        logits.append(logit)
        return logits


def log_bernoulli_prob(logits, input):
    return torch.sum(logits * input, dim=-1) - torch.sum(softplus(logits), dim=-1)


def gumbel_sigmoid(logits, temperature, hard=True, eps=1e-20):
    u = torch.rand_like(logits)
    logistic_noise = torch.log(u + eps) - torch.log(1 - u + eps)
    y = logits + logistic_noise
    # sampled = torch.sigmoid(y / temperature)
    sampled = torch.clamp((y + 1.) / (2. * temperature), 0., 1.)
    if not hard:
        return sampled
    hard_samples = (sampled > 0.5).float()
    return (hard_samples - sampled).detach() + sampled


def gumbel_softmax(logits, temperature, hard=False, eps=1e-20):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    sampled = softmax(y / temperature, dim=-1)

    if not hard:
        return sampled.view(-1, logits.shape[-1] * logits.shape[-2])

    shape = sampled.size()
    _, ind = sampled.max(dim=-1)
    y_hard = torch.zeros_like(sampled).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - sampled).detach() + sampled
    return y_hard.view(-1, logits.shape[-1] * logits.shape[-2])
