from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np

def rand_voi(segmentation, gt):

    for d in range(segmentation.ndim):
        assert segmentation.shape[d] == gt.shape[d], (
                "shapes between segmentation and gt don't match")

    return rand_voi_wrapper(
        np.ravel(segmentation, order='A'),
        np.ravel(gt, order='A'))

def rand_voi_wrapper(
        np.ndarray[uint64_t] segmentation,
        np.ndarray[uint64_t] gt):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not segmentation.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous segmentation arrray (avoid this by passing C_CONTIGUOUS arrays)")
        segmentation = np.ascontiguousarray(segmentation)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        gt = np.ascontiguousarray(gt)

    cdef uint64_t* segmentation_data
    cdef uint64_t* gt_data

    segmentation_data = <uint64_t*>segmentation.data
    gt_data = <uint64_t*>gt.data

    return rand_voi_arrays(
        segmentation.size,
        gt_data,
        segmentation_data)

cdef extern from "impl/rand_voi.hpp":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    Metrics rand_voi_arrays(
            size_t          size,
            const uint64_t* gt_data,
            const uint64_t* segmentation_data);
