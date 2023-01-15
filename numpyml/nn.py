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

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, state):
        return 1 - state**2

class RNNCell:
    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.activation = Tanh()
        bound = np.sqrt(1/hidden_size)
        self.weight_ih = np.random.uniform(low=-bound, high=bound, size=(hidden_size, input_size))
        self.weight_hh = np.random.uniform(low=-bound, high=bound, size=(hidden_size, hidden_size))
        self.bias_ih = np.random.uniform(low=-bound, high=bound, size=hidden_size,)
        self.bias_hh = np.random.uniform(low=-bound, high=bound, size=hidden_size,)

        self.grad_weight_ih = np.zeros((hidden_size, input_size))
        self.grad_weight_hh = np.zeros((hidden_size, hidden_size))
        
        self.grad_bias_ih = np.zeros(hidden_size)
        self.grad_bias_hh = np.zeros(hidden_size)

    def forward(self, x_t: np.array, h_prev: np.array) -> np.array:
        y_t = x_t@self.weight_ih.T + self.bias_ih + h_prev@self.weight_hh.T + self.bias_hh
        h_t = self.activation.forward(y_t)
        return h_t

    def backward(self, grad: np.array, h_t: np.array, h_prev_l: np.array, h_prev_t: np.array) -> Tuple[np.array]:
        dy_t = self.activation.backward(state=h_t)*grad

        self.grad_weight_ih += dy_t.T @ h_prev_l
        self.grad_weight_hh += dy_t.T @ h_prev_t
        self.grad_bias_ih += np.sum(dy_t, axis=0)
        self.grad_bias_hh += np.sum(dy_t, axis=0)

        dx = dy_t @ self.weight_ih
        dh = dy_t @ self.weight_hh

        return (dx, dh)

class RNN:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int=2) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TO DO: Check
        self.layers = [RNNCell(input_size=input_size, hidden_size=hidden_size)]
        for layer in range(num_layers-1):
            # TO DO: Check
            self.layers.append(RNNCell(input_size=hidden_size, hidden_size=hidden_size))
    
    def forward(self, x: np.array, h_0: np.array=None) -> np.array:
        batch_size, seq_len, _ = x.shape
        hiddens = np.zeros((seq_len+1, self.num_layers, batch_size, self.hidden_size))
        if h_0 is not None:
            hiddens[0, :, :, :] = h_0
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                x_t = self.layers[layer].forward(x_t, hiddens[t, layer, :, :])
                hiddens[t+1, layer, :, :] = x_t

        self.x = x
        self.hiddens = hiddens
        return hiddens[1:, -1, :, :].transpose(1, 0, 2), hiddens[-1, :, :, :]

    def backward(self, grad: np.array) -> Tuple[np.array]:
        batch_size, seq_len, input_size = self.x.shape

        dx = np.zeros((batch_size, seq_len, input_size))
        dh_0 = np.zeros((self.num_layers, batch_size, self.hidde_size))
        dh_0[-1, :, :] = grad

        for t in reversed(range(1, seq_len+1)):
            for l in reversed(range(1, self.num_layers)):
                dx_t_l, dh_0[l] = self.layers[l].backward(dh_0[l], self.hiddens[t][l], self.hiddens[t][l-1], self.hiddens[t-1][l])
                dh_0[l-1] += dx_t_l
            dx_t, dh_0[0] = self.layers[0].backward(dh_0[0], self.hiddens[t][0], self.x[:, t-1, :], self.hiddens[t-1][0])
            dx[:, t-1, :] = dx_t
        return (dx, dh_0)

        