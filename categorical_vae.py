import torch
from torch import nn


class CategoricalVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, use_nonlinear: bool = True, use_conv=False):
        super(CategoricalVAE, self).__init__()
        num_layers = len(hidden_layers)
        hidden_layers = [input_dim] + hidden_layers
        self.encode_net = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(num_layers)])
        hidden_layers = hidden_layers[::-1]
        self.decode_net = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(num_layers)])
        if use_nonlinear:
            self.generative_net = NonlinearGenerativeNet(dim_obs, dim_hids[0])
            self.inference_net = NonlinearInferenceNet(dim_obs, dim_hids[0])
        else:
            self.generative_net = GenerativeNet(
                *[nn.Linear(gen_layers[i], gen_layers[i + 1]) for i in range(len(gen_layers) - 1)])
            self.inference_net = InferenceNet(
                *[nn.Linear(inf_layers[i], inf_layers[i + 1]) for i in range(len(inf_layers) - 1)])
    def encoder(self, x):
        pass

    def decoder(self, z):
        pass

    def forward(self, x):
        pass
