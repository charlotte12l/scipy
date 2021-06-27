import numpy as np
import builtins

#pythran export calc_bin_more_pythran2(int, int[:], int[:], float[:,:])
def calc_bin_more_pythran2(Vdim, binnumbers, nbin, values):
    flatcount = np.bincount(binnumbers, None)
    a = flatcount.nonzero()
    result = np.empty([Vdim, nbin.prod()], float)
    for vv in builtins.range(Vdim):
        flatsum = np.bincount(binnumbers, values[vv])
        for i in a:
            result[vv, i] = 1 / flatcount[i]
            # result[vv, i] = flatsum[i] / flatcount[i]

    return result, flatsum