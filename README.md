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

#### Notes

* The tests compare this implementation with the output of PyTorch's `Conv2d` layer and might trigger the following warning, which is caused by PyTorch and can be safely ignored: `/home/user/.local/lib/python3.10/site-packages/torch/nn/modules/conv.py:459: UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created (Triggered internally at ../aten/src/ATen/native/Convolution.cpp:1003.)
  return F.conv2d(input, weight, bias, self.stride,`
* This function actually computes a correlation instead of a convolution since the kernel is not flipped (same as PyTorch and other popular deep learning frameworks). If you want a "real" convolution, you can flip the kernel weights: `weight[:, :, ::-1, ::-1]`
* If you want a *fast* `conv2d` implementation, use PyTorch instead. The main purpose of *this* implementation is that it is lightweight.
