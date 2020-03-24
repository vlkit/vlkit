# =================================================
# * Licensed under The MIT License
# * Written by KAI-ZHAO and Shanghua-Gao
# =================================================
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "nms.h":
  void nms (float * in_array1, float * in_array2, float * out_array, int h, int w)

# create the wrapper code, with numpy type annotations
def nms_func(np.ndarray[float, ndim=2, mode="c"] in_array1 not None,
             np.ndarray[float, ndim=2, mode="c"] in_array2 not None,
             np.ndarray[float, ndim=2, mode="c"] out_array not None):
  assert in_array1.shape[0] == out_array.shape[0] and in_array1.shape[0] == in_array2.shape[0]
  assert in_array1.shape[1] == out_array.shape[1] and in_array1.shape[1] == in_array2.shape[1]
  nms(<float*> np.PyArray_DATA(in_array1),
      <float*> np.PyArray_DATA(in_array2),
      <float*> np.PyArray_DATA(out_array),
      in_array1.shape[0], in_array1.shape[1])
