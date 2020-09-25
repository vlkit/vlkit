import cv2
import numpy as np
from PIL import Image
from os.path import isfile
from .image import isimg, norm255

def imread(path, backend="pil", grayscale=False):
    assert isfile(path) and isimg(path)
    if backend == "pil":
        im = Image.open(path)
        if grayscale:
            im = im.convert("L")
        return np.array(im)
    elif backend == "cv2":
        return cv2.imread(path, int(not grayscale))
    else:
        raise ValueError("Invalid backend %s. (Valid options are ['pil', 'cv2'])" % backend)

def imwrite(im, path, backend="pil", normalize=False):
    assert isinstance(im, np.ndarray)
    assert isimg(path)
    if normalize:
        im = norm255(im)
    if backend == "pil":
        Image.fromarray(im).save(path)
    elif backend == "cv2":
        cv2.imwrite(path, im)
    else:
        raise ValueError("Invalid backend %s. (Valid options are ['pil', 'cv2'])" % backend)
