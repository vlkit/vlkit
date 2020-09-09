import cv2
import numpy as np

def seg2edge(seg):
    grad = np.gradient(seg.astype(np.float32))
    grad = np.sqrt(grad[0]**2 + grad[1]**2)
    return grad != 0

def sobel(x, kernel_size=3):
    sobelx = cv2.Sobel(dist,cv2.CV_64F,1,0,ksize=9)
    sobely = cv2.Sobel(dist,cv2.CV_64F,0,1,ksize=9)

    sobel = np.sqrt(sobelx**2 + sobely**2)
    return sobel, sobely, sobelx

def flux2angle(flux):
    """
    flux: a [2, H, W] tesnro represents the flux vector of each position
    """
    _, H, W = flux.shape
    top_half = flux[0,...] >= 0 # y >= 0, \theta <= \pi
    bottom_half = flux[0,...] < 0 # y < 0, \theta > \pi

    # unit vector [y=0, x=1]
    unit = np.zeros((2, H, W), dtype=np.float32)
    unit[1,...] = 1

    cos = (flux * unit).sum(axis=0)
    acos = np.arccos(cos)
    angle = np.zeros((H, W), dtype=np.float32)
    angle[top_half] = acos[top_half]
    angle[bottom_half] = 2*np.pi - acos[bottom_half]

    return angle
