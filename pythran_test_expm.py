import numpy as np
import scipy
import math
from timeit import timeit

print(scipy.__file__)

def random_sparse_csc(m, n, nnz_per_row, rng):
    # Copied from the scipy.sparse benchmark.
    rows = np.arange(m).repeat(nnz_per_row)
    cols = rng.integers(0, n, size=nnz_per_row*m)
    vals = rng.random(m*nnz_per_row)
    M = scipy.sparse.coo_matrix((vals, (rows, cols)), (m, n), dtype=float)
    # Use csc instead of csr, because sparse LU decomposition
    # raises a warning when I use csr.
    return M.tocsc()

rng = np.random.default_rng(1234)
n = 100
# Let the number of nonzero entries per row
# scale like the log of the order of the matrix.
nnz_per_row = int(math.ceil(math.log(n)))

# time the sampling of a random sparse matrix
A_sparse = random_sparse_csc(n, n, nnz_per_row, rng)

# first format conversion
A_dense = A_sparse.toarray()

f1 = lambda: scipy.linalg.expm(A_sparse)
f2 = lambda: scipy.linalg.expm(A_dense)

print(timeit(f1, number=10)) 
print(timeit(f2, number=10)) 