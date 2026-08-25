import torch
import torch.nn as nn


class MLP(nn.Module):
    """Generic fully connected multilayer perceptron. Default arguments used in smiley example."""
    def __init__(
        self,
        inDim=4,
        hiddenDims=(256, 256, 256, 256),
        outDim=3,
        activation=nn.SiLU
    ):
        super().__init__()

        dims = (inDim,) + tuple(hiddenDims) + (outDim,)

        layers = []

        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                activation()
            ]

        layers += [
            nn.Linear(dims[-2], dims[-1])
        ]

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)