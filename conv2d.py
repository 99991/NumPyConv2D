import numpy as np

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
    """
    Applies a 2D convolution to an array of images. Technically, this function
    computes a correlation instead of a convolution since the kernels are not
    flipped.

    input: numpy array of images with shape = (n, c, h, w)
    weight: numpy array with shape = (c_out, c // groups, kernel_height, kernel_width)
    bias: numpy array of shape (c_out,), default None
    stride: step width of convolution kernel, int or (int, int) tuple, default 1
    padding: padding around images, int or (int, int) tuple or "same", default 0
    dilation: spacing between kernel elements, int or (int, int) tuple, default 1
    groups: split c and c_out into groups to reduce memory usage (must both be divisible), default 1
    padding_mode: "zeros", "reflect", "replicate", "circular", or whatever np.pad supports, default "zeros"

    This function is indended to be similar to PyTorch's conv2d function.
    For more details, see:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """
    c_out, c_in_by_groups, kh, kw = weight.shape
    kernel_size = (kh, kw)

    if isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    if padding:
        input = conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size)

    n, c_in, h, w = input.shape
    dh, dw = dilation
    sh, sw = stride
    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1
    assert c_in % groups == 0
    assert c_out % groups == 0
    c_in_group = c_in // groups
    c_out_group = c_out // groups
    kernel_shape = (c_in_group, dilated_kh, dilated_kw)

    input = input.reshape(n, groups, c_in_group, h, w)
    weight = weight.reshape(groups, c_out_group, c_in_by_groups, kh, kw)

    # Cut out kernel-shaped regions from input
    windows = np.lib.stride_tricks.sliding_window_view(input, kernel_shape, axis=(-3, -2, -1))

    # Apply stride and dilation. Prepare for broadcasting to handle groups.
    windows = windows[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
    weight = weight[np.newaxis, :, :, np.newaxis, np.newaxis, :, :, :]
    h_out, w_out = windows.shape[3:5]

    # Dot product equivalent to either of the next two lines but 10 times faster
    #y = np.einsum("abcdeijk,abcdeijk->abcde", windows, weight)
    #y = (windows * weight).sum(axis=(-3, -2, -1))
    windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
    weight = weight.reshape(1, groups, c_out_group, 1, 1, c_in_group * kh * kw)
    y = np.einsum("abcdei,abcdei->abcde", windows, weight)

    # Concatenate groups as output channels
    y = y.reshape(n, c_out, h_out, w_out)

    if bias is not None:
        y = y + bias.reshape(1, c_out, 1, 1)

    return y

def conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size):
    if padding == "valid":
        return input

    if padding == "same":
        h, w = input.shape[-2:]
        sh, sw = stride
        dh, dw = dilation
        kh, kw = kernel_size
        ph = (h - 1) * (sh - 1) + (kh - 1) * dh
        pw = (w - 1) * (sw - 1) + (kw - 1) * dw
        ph0 = ph // 2
        ph1 = ph - ph0
        pw0 = pw // 2
        pw1 = pw - pw0
    else:
        if isinstance(padding, int):
            padding = [padding, padding]
        ph0, pw0 = padding
        ph1, pw1 = padding

    pad_width = ((0, 0), (0, 0), (ph0, ph1), (pw0, pw1))

    mode = {
        "zeros": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }.get(padding_mode, padding_mode)

    return np.pad(input, pad_width, mode)
