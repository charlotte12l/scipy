import numpy as np
import scipy.stats as stats
from timeit import timeit

rng = np.random.default_rng(12345678)
inp = rng.random(9999).reshape(3, 3333) * 200
subbin_x_edges = np.arange(0, 200, dtype=np.float32)
subbin_y_edges = np.arange(0, 200, dtype=np.float64)


f_min = lambda: stats.binned_statistic_dd(
    [inp[0], inp[1]], inp[2], statistic='min',
    bins=[subbin_x_edges, subbin_y_edges])

f_max = lambda: stats.binned_statistic_dd(
    [inp[0], inp[1]], inp[2], statistic='max',
    bins=[subbin_x_edges, subbin_y_edges])

f_std = lambda: stats.binned_statistic_dd(
    [inp[0], inp[1]], inp[2], statistic='std',
    bins=[subbin_x_edges, subbin_y_edges])

f_median = lambda: stats.binned_statistic_dd(
    [inp[0], inp[1]], inp[2], statistic='median',
    bins=[subbin_x_edges, subbin_y_edges])

print(timeit(f_min, number=100)) # 1.3672322449999998 vs 0.40702188699999997(python vs pythran)
print(timeit(f_max, number=100)) # 1.346275941 vs 0.496832677
print(timeit(f_std, number=100)) # 0.47545129600000013 vs 0.3890081329999999
print(timeit(f_median, number=100)) # 5.6123907740000005 vs 0.4219133049999999