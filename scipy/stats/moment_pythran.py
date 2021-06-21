import numpy as np

#pythran export _moment(float[:], int[:], int, None)
#pythran export _moment(float[:], int[:], None, None)
#pythran export _moment(int[:], int[:], int, None)
#pythran export _moment(int[:], int[:], None, None)
#pythran export _moment(float[:], int, int, None)
#pythran export _moment(float[:], int, None, None)
#pythran export _moment(int[:], int, int, None)
#pythran export _moment(int[:], int, None, None)
def _moment(a, moment, axis, mean=None):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    if moment == 0 or moment == 1:
        # By definition the zeroth moment about the mean is 1, and the first
        # moment is 0.
        shape = list(a.shape)
        del shape[axis]
        if ((a.dtype == np.float64) or (a.dtype == np.float32) \
        or (a.dtype == np.float128) or np.iscomplex(a)):
            dtype = a.dtype.type  
        else:
            dtype = np.float64

        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return (np.ones(shape, dtype=dtype) if moment == 0
                    else np.zeros(shape, dtype=dtype))
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = np.mean(a, axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean**2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return np.mean(s, axis)



'''
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
'''
'''
#pythran export _calc_binned_statistic_pythran(int, int[:], float[:,:], float[:,:], string)
def _calc_binned_statistic_pythran(Vdim, bin_numbers, result, values, stat_func):
    unique_bin_numbers = np.unique(bin_numbers)
    for vv in builtins.range(Vdim):
        bin_map = _create_binned_data(bin_numbers, unique_bin_numbers,
                                      values, vv)
        for i in unique_bin_numbers:
            # if the stat_func is callable, all results should be updated
            # if the stat_func is np.std, calc std only when binned data is 2
            # or more for speed up.
            if stat_func == 'std':
                result[vv, i] = np.std(np.array(bin_map[i]))
            elif stat_func == 'median':
                result[vv, i] = np.median(np.array(bin_map[i]))
            elif stat_func == 'min':
                result[vv, i] = np.min(np.array(bin_map[i]))
            elif stat_func == 'max':
                result[vv, i] = np.min(np.array(bin_map[i]))
            else:
                raise Exception('Exception: {stat_func} is not supported')
'''