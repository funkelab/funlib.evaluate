from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.map cimport map as cpp_map
import numpy as np
cimport numpy as np

def find_centers_cpp(np.ndarray[uint64_t, ndim=3] labels):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not labels.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous label arrray (avoid this by passing C_CONTIGUOUS arrays)")
        labels = np.ascontiguousarray(labels)

    cdef uint64_t* labels_data
    labels_data = <uint64_t*>labels.data

    return centers(
        labels.shape[0],
        labels.shape[1],
        labels.shape[2],
        labels_data)

cdef extern from "impl/centers.hpp":

    struct Center:
        double z
        double y
        double x

    cpp_map[uint64_t, Center] centers(
            size_t size_z,
            size_t size_y,
            size_t size_x,
            const uint64_t* labels);

