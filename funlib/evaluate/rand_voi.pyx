from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np

def rand_voi(truth, test):

    for d in range(truth.ndim):
        assert truth.shape[d] == test.shape[d], (
                "shapes between truth and test don't match")

    return rand_voi_wrapper(
        np.ravel(truth, order='A'),
        np.ravel(test, order='A'))

def rand_voi_wrapper(
        np.ndarray[uint64_t] truth,
        np.ndarray[uint64_t] test):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if truth is not None and not truth.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        truth = np.ascontiguousarray(truth)
    if not test.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous test arrray (avoid this by passing C_CONTIGUOUS arrays)")
        test = np.ascontiguousarray(test)

    cdef uint64_t* test_data
    cdef uint64_t* truth_data

    test_data = <uint64_t*>test.data
    truth_data = <uint64_t*>truth.data

    return rand_voi_arrays(
        test.size,
        truth_data,
        test_data)

cdef extern from "impl/rand_voi.hpp":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    Metrics rand_voi_arrays(
            size_t          size,
            const uint64_t* truth_data,
            const uint64_t* test_data);
