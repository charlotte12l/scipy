import numpy as np

#pythran export calc_test(int[:])
def calc_test(binnumbers):
    return np.bincount(binnumbers, None)