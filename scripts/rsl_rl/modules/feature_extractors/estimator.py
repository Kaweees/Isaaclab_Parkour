import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class DefaultEstimator(nn.Module):
    def __init__(self, num_prop, num_priv_explicit, hidden_dims=[256, 128, 64], activation="elu", **kwargs):
        super().__init__()

        self.input_dim = num_prop
        self.output_dim = num_priv_explicit
        activation = resolve_nn_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for layer in range(len(hidden_dims)):
            if layer == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[layer], num_priv_explicit))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))
                estimator_layers.append(activation)
        self.estimator = nn.Sequential(*estimator_layers)

    def forward(self, input):
        return self.estimator(input)

    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)
