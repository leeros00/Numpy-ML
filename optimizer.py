import numpy as np
from . import nn

class SGD:
    def __init__(self, model: nn.Sequential, lr: float) -> None:
        self.layers = model.layers
        self.lr = lr

    def zero_grad(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.grad_weight.fill(0)
                layer.grad_bias.fill(0)

    def step(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight = layer.weight - self.lr*layer.grad_weight
                layer.bias = layer.bias - self.lr*layer.grad_bias

                