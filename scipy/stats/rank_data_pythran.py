import numpy as np


#pythran export rankdata_pythran(float[:], string, None)
#pythran export rankdata_pythran(float[:], string, int)
def rankdata_pythran(a, method='average', axis=None):
    if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
        raise ValueError('unknown method "{0}"'.format(method))

    # if axis is not None:
    #     a = np.asarray(a)
    #     if a.size == 0:
    #         # The return values of `normalize_axis_index` are ignored.  The
    #         # call validates `axis`, even though we won't use it.
    #         # use scipy._lib._util._normalize_axis_index when available
    #         np.core.multiarray.normalize_axis_index(axis, a.ndim)
    #         dt = np.float64 if method == 'average' else np.int_
    #         return np.empty(a.shape, dtype=dt)
    #     return np.apply_along_axis(rankdata_pythran, axis, a, method)

    arr = np.ravel(np.asarray(a))
    algo = 'mergesort' if method == 'ordinal' else 'quicksort'
    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    if method == 'ordinal':
        return inv + 1

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    if method == 'dense':
        return dense

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    if method == 'max':
        return count[dense]

    if method == 'min':
        return count[dense - 1] + 1

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)