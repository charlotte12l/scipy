import numpy as np
import builtins


#pythran export _create_binned_data_pythran(int[:], int[:], float[:,:], int)
def _create_binned_data_pythran(bin_numbers, unique_bin_numbers, values, vv):
    """ Create hashmap of bin ids to values in bins
    key: bin number
    value: list of binned data
    """
    bin_map = dict()
    for i in unique_bin_numbers:
        bin_map[i] = []
    for i in builtins.range(len(bin_numbers)):
        bin_map[bin_numbers[i]].append(values[vv, i])
    return bin_map



#pythran export _calc_bin_more_pythran(int, int[:], int[:], int[:,:], str)
def _calc_bin_more_pythran(Vdim, binnumbers, nbin, values, statistic):

    result = np.empty([Vdim, nbin.prod()], float)

    if statistic == 'mean':
        result.fill(np.nan)
        flatcount = np.bincount(binnumbers)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            for i in a:
                result[vv, i] = flatsum[i] / flatcount[i]
    elif statistic == 'std':
        result.fill(0)
        unique_bin_numbers = np.unique(binnumbers)
        for vv in builtins.range(Vdim):
            bin_map = _create_binned_data_pythran(binnumbers, unique_bin_numbers,
                                        values, vv)
            for i in unique_bin_numbers:
                result[vv, i] = np.std(np.array(bin_map[i]))
    elif statistic == 'count':
        result.fill(0)
        flatcount = np.bincount(binnumbers)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
        # flatcount = np.expand_dims(flatcount, axis=0)
        # flatcount_2d = flatcount[np.newaxis, :]
        # for i in a:
        #     result[0, i] = flatcount[i]
    elif statistic == 'sum':
        result.fill(0)
        for vv in builtins.range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            for i in a:
                result[vv, i] = flatsum[i]
    elif statistic == 'median':
        result.fill(np.nan)
        unique_bin_numbers = np.unique(binnumbers)
        for vv in builtins.range(Vdim):
            bin_map = _create_binned_data_pythran(binnumbers, unique_bin_numbers,
                                        values, vv)
            for i in unique_bin_numbers:
                result[vv, i] = np.median(np.array(bin_map[i]))
    elif statistic == 'min':
        result.fill(np.nan)
        unique_bin_numbers = np.unique(binnumbers)
        for vv in builtins.range(Vdim):
            bin_map = _create_binned_data_pythran(binnumbers, unique_bin_numbers,
                                        values, vv)
            for i in unique_bin_numbers:
                result[vv, i] = np.min(np.array(bin_map[i]))
    elif statistic == 'max':
        result.fill(np.nan)
        unique_bin_numbers = np.unique(binnumbers)
        for vv in builtins.range(Vdim):
            bin_map = _create_binned_data_pythran(binnumbers, unique_bin_numbers,
                                        values, vv)
            for i in unique_bin_numbers:
                result[vv, i] = np.max(np.array(bin_map[i]))
    else:
        pass      
