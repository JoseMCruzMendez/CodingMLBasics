import numpy as np
from NpAutograd import Tensor, GradModule
from typing import List, Callable, Union, Literal, Tuple
import NpAutograd as npa

class ConvLayer(GradModule):
    def __init__(self, kernel_size, in_channels = 1, out_channels=1, stride=1, padding:int=0, activation:Callable[[npa.Tensor], npa.Tensor] = npa.sigmoid):
        """Implements a basic convolutional layer.
        :param kernel_size: size of the convolutional kernel, will be a kernel of shape (kernel_size, kernel_size)
        :param stride: stride of the pool layers, defaults to 1
        :param padding: padding applied to layer inputs.
        :param activation: Activation function to apply after convolution, defaults to sigmoid.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size//2 <= padding, "Padding must be more than half the kernel size"
        self.padding = padding

        #Since I will only be using this on MNIST all channels will be assumed to be 1.
        #Furthermore, I will assume all kernels to be square
        self.kernel = Tensor(np.random.randn(1, out_channels, in_channels, kernel_size, kernel_size)) #(broadcast_dim, C_out, C_in, k_h, k_w)
        self.bias = Tensor(np.random.randn(self.out_channels))
        self.params = [self.kernel, self.bias]
        self.activation = activation



    def forward(self, value: Tensor, *args, **kwargs):
        #tensor must be of shape (batch, channels, height, width)
        if isinstance(value, np.ndarray):
            value = Tensor(value)
        assert value.v.shape[1] == self.in_channels, "Input channels must match kernel channels"
        batch, channels, height, width = value.v.shape
        padded_v = np.pad(value.v, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        loop_height = (height - self.kernel_size + 2 * self.padding)
        out_height = loop_height // self.stride + 1
        loop_width = (width - self.kernel_size + 2 * self.padding)
        out_width = loop_width // self.stride + 1
        res = np.zeros((batch, self.out_channels, out_height, out_width))
        out_i, out_j = 0, 0
        for i in range(0, loop_height + 1, self.stride):
            for j in range(0, loop_width + 1, self.stride):
                patch = padded_v[:, None, :, i:i+self.kernel_size, j:j+self.kernel_size] * self.kernel.v #shape (batch, C_out, C_in, k_h, k_w)
                #Since kernel is padded, the leading broadcast_dim allows broadcasting over batchsize, and the None in padded_v corresponds to C_out broadcasting
                res[:, :, out_i, out_j] = np.sum(patch, axis=(2,3,4)) + self.bias.v #now we sum over C_in, k_h, k_w to obtain a (batch, C_out) result.
                #Since we're accumulating in res, we obtain a (batch, C_out, h_out, w_out) tensor
                out_j +=1
            out_j = 0
            out_i +=1
        out = Tensor(res)
        out._prev = [value]
        def _backward():
            grad_padded = np.zeros((batch, channels, height + 2*self.padding, width + 2*self.padding))
            kernel_grad = np.zeros_like(self.kernel.v)     # (C_out, C_in, K, K)
            bias_grad   = np.zeros_like(self.bias.v)       # (C_out,)
            for i in range(out_height):
                r = i * self.stride
                for j in range(out_width):
                    c = j * self.stride
                    conv_patch = padded_v[:, :, r:r+self.kernel_size, c:c+self.kernel_size]   # shape (batch, C_in, K, K)
                    g = out.grad[:,:,i,j] #(batch, C_out)
                    bias_grad += g.sum(axis=0) #sum over batch and get C_out
                    kernel_grad += np.einsum('bo,bilw->oilw', g, conv_patch)[None,:,:,:,:] #sums g[::NNN]*patch[:N:::] over ax 0, then adds broadcast_dim
                    # multiply by each filter’s weights:
                    # this yields shape (batch, out_channels, in_channels, K, K)
                    # then sum over out_channels to collapse into in_channels
                    # → (batch, in_channels, K, K)
                    window_grad = np.sum(
                        self.kernel.v[:,:, :, :, :] * g[:, :, None, None, None],
                        axis=1
                    )#(1,C_out,C_in,k_h,k_w) x (batch, C_out, 1,1,1) summed over C_out

                    # now add that into the correct slice of grad_padded
                    grad_padded[:, :, r:r+self.kernel_size, c:c+self.kernel_size] += window_grad
            if self.padding > 0:
                value.grad += grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                value.grad += grad_padded
            self.kernel.grad += kernel_grad
            self.bias.grad += bias_grad
        out._backward = _backward
        act = self.activation(out)
        return act

def Pool(kernel_size, stride=1, type:Literal["max", "avg"]="max"):
    pool_func = np.max if type == "max" else np.mean
    def max_pool(value: Tensor):
        #Size prep for loops
        batch, channels, height, width = value.v.shape
        loop_height = (height - kernel_size)
        out_height = loop_height//stride + 1
        loop_width = (width - kernel_size)
        out_width = loop_width //stride  + 1
        res = np.zeros((batch, channels, out_height, out_width))
        out_i, out_j = 0, 0
        #Actual convolution
        for i in range(0, loop_height + 1, stride):
            for j in range(0, loop_width + 1, stride):
                #Pools over width and height, leaving batch and channel untouched
                patch = pool_func(value.v[:, :, i:i+kernel_size, j:j+kernel_size], axis=(2,3))
                res[:, :, out_i, out_j] = patch
                out_j +=1
            out_j = 0
            out_i += 1
        out = Tensor(res)
        out._prev = [value]
        #Bacwards + returning out
        def _backward():
            g = np.zeros_like(value.v)
            for i in range(out_height):
                i_val = i * stride
                for j in range(out_width):
                    j_val = j * stride
                    patch = out.v[:, :, i:i+kernel_size, j:j+kernel_size]  # (batch, C, p_h, p_w)
                    g_patch = out.grad[:, :, i, j]                # (batch, C)
                    if type == "max":
                        # build a mask of where the max occurred
                        max_vals = patch.max(axis=(2,3))            # (batch, C)
                        mask = (patch == max_vals[:, :, None, None])  # (batch, C, p_h, p_w)
                        # scatter only to those positions
                        g[:, :, i_val:i_val+kernel_size, j_val:j_val+kernel_size] += mask * g_patch[:, :, None, None]

                    else:  # average pool
                        # spread gradient evenly
                        g[:, :, i_val:i_val + kernel_size, j_val:j_val + kernel_size] += g_patch[:, :, None, None] / (kernel_size ** 2)
            value.grad += g
        out._backward = _backward
        return out
    return max_pool


