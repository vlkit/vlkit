import torch, math
import torch.nn as nn
import numpy as np

def upsample_filter(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    reference: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    weights = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    return torch.tensor(weights)

def deconv_upsample(channels, stride, fixed=True):
    """
    channels: number of input/output channels
    stride: upsampling factor
    fixed: whether fix deconv parameters (default: True)
    """
    assert stride % 2 == 0
    padding = stride // 2
    kernel_size = stride * 2

    upsample = nn.ConvTranspose2d(channels, channels, kernel_size, stride=stride,
      padding=padding,output_padding=0, groups=channels,bias=False)
    upsample.weight.data.copy_(upsample_filter(kernel_size))

    if fixed:
        upsample.weight.requires_grad = False

    return upsample