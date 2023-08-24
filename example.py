from conv2d import conv2d
import numpy as np

# Number of input images, channels, height, width, output channels, kernel size
n, c, h, w, c_out, kernel_size = 5, 3, 32, 32, 4, 3

# Random input image batch and kernel weights
input = np.random.rand(n, c, h, w)
weight = np.random.rand(c_out, c, kernel_size, kernel_size)

# Simple example usage of conv2d convolution
output = conv2d(input, weight)

# Advanced usage with all parameters
output = conv2d(input, weight, bias=np.random.rand(c_out), stride=(2, 2),
    padding=(0, 0), dilation=(2, 2), groups=1, padding_mode="replicate")

