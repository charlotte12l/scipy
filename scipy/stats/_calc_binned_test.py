import builtins
import numpy as np

#pythran export _calc_binned_statistic_median(int, int64[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_median(int, int64[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_median(int, intc[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_median(int, intc[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_median(int, int[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_median(int, int[:], float[:,:], int[:,:])
def _calc_binned_statistic_median(Vdim, bin_numbers, result, values):
    for vv in builtins.range(Vdim):
        i = np.lexsort((values[vv], bin_numbers))
        _, j, counts = np.unique(bin_numbers[i],
                                    return_index=True, return_counts=True)
        mid = j + (counts - 1) / 2
        mid_a = values[vv, i][np.floor(mid).astype(int)]
        mid_b = values[vv, i][np.ceil(mid).astype(int)]
        medians = (mid_a + mid_b) / 2
        result[vv, bin_numbers[i][j]] = medians


#pythran export _calc_binned_statistic_min(int, int64[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_min(int, int64[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_min(int, intc[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_min(int, intc[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_min(int, int[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_min(int, int[:], float[:,:], int[:,:])
def _calc_binned_statistic_min(Vdim, bin_numbers, result, values):
    for vv in builtins.range(Vdim):
        i = np.argsort(values[vv])[::-1]  # Reversed so the min is last
        result[vv, bin_numbers[i]] = values[vv, i]

#pythran export _calc_binned_statistic_max(int, int64[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_max(int, int64[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_max(int, intc[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_max(int, intc[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_max(int, int[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_max(int, int[:], float[:,:], int[:,:])
def _calc_binned_statistic_max(Vdim, bin_numbers, result, values):
    for vv in builtins.range(Vdim):
        i = np.argsort(values[vv])
        result[vv, bin_numbers[i]] = values[vv, i]


#pythran export _calc_binned_statistic_std(int, int64[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_std(int, int64[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_std(int, intc[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_std(int, intc[:], float[:,:], int[:,:])
#pythran export _calc_binned_statistic_std(int, int[:], float[:,:], float[:,:])
#pythran export _calc_binned_statistic_std(int, int[:], float[:,:], int[:,:])
def _calc_binned_statistic_std(Vdim, bin_numbers, result, values):
    for vv in builtins.range(Vdim):
        i = np.argsort(values[vv])
        result[vv, bin_numbers[i]] = values[vv, i]
        flatcount = np.bincount(bin_numbers)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = np.bincount(bin_numbers, values[vv])
            delta = values[vv] - flatsum[bin_numbers] / flatcount[bin_numbers]
            std = np.sqrt(np.bincount(bin_numbers, delta**2)[a] / flatcount[a])
            result[vv, a] = std