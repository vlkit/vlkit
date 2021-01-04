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

class ArcFace(nn.Module):
    """ 
    ArcFace https://arxiv.org/pdf/1801.07698
    """
    def __init__(self, in_features, out_features, s=32, m=0.5):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.s = s 
        self.m = m 
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None or self.m == 0:
            return cosine

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) 

        # psi = cos(theta + m)
        psi_theta = cosine*self.cos_m - sine*self.sin_m
        psi_theta = torch.where(cosine > -self.cos_m, psi_theta, -psi_theta-2)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot*psi_theta + (1-one_hot)*cosine) * self.s

        return output, cosine.detach() * self.s

    def __str__(self):
        return "ArcFace() in_features=%d out_features=%d s=%.3f m=%.3f" % \
               (self.weight.shape[1], self.weight.shape[0], self.s, self.m)
