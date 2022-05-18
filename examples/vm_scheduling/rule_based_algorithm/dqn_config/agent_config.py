# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import RMSprop

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, PolicyGradient, SimpleMultiHeadModel

n_servers = 25

input_dim = (
    2 * n_servers * 3 + 3
)

agent_config = {
    "model": {
        "input_dim": input_dim,
        "output_dim": n_servers,   # number of possible actions
        "hidden_dims": [256, 128, 64],
        "activation": nn.LeakyReLU,
        "softmax": False,
        "batch_norm": True,
        "skip_connection": False,
        "head": True,
        "dropout_p": .0
    },
    "optimization": OptimOption(optim_cls=RMSprop, optim_params={"lr": 0.01}),
    "hyper_params": {
        "reward_discount": .0,
        "loss_cls": nn.SmoothL1Loss,
        "target_update_freq": 5,
        "tau": 0.1,
        "double": False
    }
}
