from conv2d import conv2d
import numpy as np

def test_conv2d(n, c_in, c_out, h, w, groups, stride, padding, dilation, kernel_size, padding_mode="zeros"):
    # Compare NumPy conv2d implementation to torch.nn.Conv2d
    import torch
    import torch.nn as nn

    input = torch.rand(n, c_in, h, w)

    conv = nn.Conv2d(
        c_in,
        c_out,
        kernel_size=kernel_size,
        bias=True,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode)

    torch_output = conv(input).detach().numpy()

    input = input.detach().numpy()
    weight = conv.weight.detach().numpy()
    bias = conv.bias.detach().cpu().numpy()

    output = conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode)

    mean_squared_error = np.mean(np.square(output - torch_output))

    assert mean_squared_error < 1e-10

def tests():
    np.random.seed(0)

    # Test padding="same"
    for w in range(30, 35):
        for kernel_size in [1, 2, 3, 4, 5]:
            for dilation in [1, 2, 3, 4]:
                test_conv2d(n=1, c_in=1, c_out=1, h=1, w=w, groups=1, stride=1, padding="same", dilation=dilation, kernel_size=kernel_size)

    # Test 100 random parameters
    for _ in range(100):
        n = np.random.randint(1, 5)
        w = np.random.randint(30, 40)
        h = np.random.randint(30, 40)
        groups = np.random.randint(1, 5)
        c_in = groups * np.random.randint(1, 5)
        c_out = groups * np.random.randint(1, 5)
        stride = (np.random.randint(1, 4), np.random.randint(1, 4))
        padding = (np.random.randint(1, 4), np.random.randint(1, 4))
        dilation = (np.random.randint(1, 4), np.random.randint(1, 4))
        kernel_size = (np.random.randint(1, 4), np.random.randint(1, 4))
        padding_mode = np.random.choice(["zeros", "reflect", "replicate", "circular"])
        test_conv2d(n, c_in, c_out, h, w, groups, stride, padding, dilation, kernel_size, padding_mode)

    print("Tests passed :)")

if __name__ == "__main__":
    tests()

