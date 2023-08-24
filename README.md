# NumPyConv2D

Sometimes, PyTorch can be a bit much when all you want is a simple 2d convolution and don't particularly care about speed.
This repository implements a fully vectorized [`conv2d`](https://github.com/99991/NumPyConv2D/blob/main/conv2d.py) function similar to PyTorch's [`torch.nn.functional.conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html) with support for stride, padding, dilation, groups and all kinds of padding.
The only dependency is NumPy.

## Example

```python
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
```
