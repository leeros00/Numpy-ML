import numpy as np
from typing import *

class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        bound = np.sqrt(6/in_features)
        self.weight = np.random.uniform(low=-bound,
                                        high=bound,
                                        size=(in_features, out_features))

        self.bias = np.zeros(np.shape(1, out_features))

        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros((1, out_features))

        self.x = None

    def forward(self, x: np.array) -> np.array:
        '''Forward pass for linear layers'''

        self.x = x

        return x@self.weight + self.bias

    def backward(self, grad: np.array) -> np.array:
        self.grad_weight = self.x.T@grad
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        return grad@self.weight.T

class ReLU:
    def __init__(self):
        self.x = None
    
    def forward(self, x: np.array) -> np.array:
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad: np.array) -> np.array:
        state = self.x
        mask = np.where(state <= 0, 0, 1)
        return grad*mask

class CrossEntropyLoss:
    def __init__(self) -> None:
        self.input = None
        self.target = None

    def forward(self, input: np.array, target: np.array) -> Union[float, np.float64]:
        target = make_one_hot(target, num_classes=input.shape[1])
        self.input = input
        self.target = target
        loss = -np.sum(target*np.log(softmax(input)), axis=1)
        return np.mean(loss)

    def backward(self) -> np.array:
        batch_size = self.input.shape[0]
        return (softmax(self.input) - self.target)/batch_size

class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = list(layers)
    
    def forward(self, x: np.array=None) -> np.array:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_function: CrossEntropyLoss) -> np.array:
        grad = loss_function.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

def softmax(x: np.array) -> np.array:
    a = np.max(x, axis=1, keepdims=True)
    denom = a + np.log(np.sum(np.exp(x - a), axis=1, keepdims=True))
    return np.exp(x - denom)

def make_one_hot(idx_labels: np.array, num_classes: Optional[int]=10) -> np.array:
    one_hot_labels = np.zeros((len(idx_labels), num_classes))
    one_hot_labels[np.arange(len(idx_labels)), idx_labels] = 1
    return one_hot_labels

## CNN stuff
def get_conv1d_output_size(input_size: int, kernel_size: int, stride: int) -> int:
    return (input_size - kernel_size) // stride + 1

class Conv1d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: Optional[int]=1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        bound = np.sqrt(1/(in_channels*kernel_size))
        self.weight = np.random.normal(-bound, bound, size=(out_channels, in_channels, kernel_size))
        self.bias = np.random.normal(-bound, bound, size=(out_channels,))

        self.grad_weight = np.zeros(self.weight.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, x: np.array) -> np.array:
        batch_size, _, input_size = x.shape
        self.x = x

        self.output_size = get_conv1d_output_size(input_size, self.kernel_size, self.stride)
        out = np.zeros([batch_size, self.out_channels, self.output_size])

        for i in range(self.output_size):
            b = i*self.stride
            e = b + self.kernel_size
            out[:, :, i] = np.tensordot(x[:, :, b:e], self.weight, axes=([1, 2], [1, 2])) + self.bias
        return out

    def backward(self, delta: np.array) -> np.array:
        dx = np.zeros(self.x.shape)
        for i in range(self.output_size):
            b = i*self.stride
            e = b + self.kernel_size
            dx[:, :, b:e] += np.tensordot(delta[:, :, i], self.weight, axes=(1))
            self.grad_weight += np.tensordot(delta[:, :, i].T, self.x[:, :, b:e], axes=(1))
        self.grad_bias = np.sum(delta, axis=(0, 2))
        return dx

        
        