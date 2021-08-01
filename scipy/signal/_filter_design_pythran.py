import numpy as np
import math


#pythran export _cplxreal(complex[:],float)
def _cplxreal(z, tol):
    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[np.lexsort((abs(z.imag), z.real))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return np.array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(np.concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[:] = chunk[np.lexsort([abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr


# TODO: Make this a real public function scipy.misc.ff
#pythran export _falling_factorial(int, int)
def _falling_factorial(x, n):
    r"""
    Return the factorial of `x` to the `n` falling.

    This is defined as:

    .. math::   x^underline n = (x)_n = x (x-1) \cdots (x-n+1)

    This can more efficiently calculate ratios of factorials, since:

    n!/m! == falling_factorial(n, n-m)

    where n >= m

    skipping the factors that cancel out

    the usual factorial n! == ff(n, n)
    """
    val = 1
    for k in range(x - n + 1, x + 1):
        val *= k
    return val


#pythran export _bessel_poly(int, bool?)
def _bessel_poly(n, reverse=False):
    """
    Return the coefficients of Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498, and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    >>> i = 0
    >>> for n in range(51):
    ...     for x in _bessel_poly(n, reverse=True):
    ...         print(i, x)
    ...         i += 1

    """
    if abs(int(n)) != n:
        raise ValueError("Polynomial order must be a nonnegative integer")
    else:
        n = int(n)  # np.int32 doesn't work, for instance

    out = []
    for k in range(n + 1):
        num = _falling_factorial(2*n - k, n)
        den = 2**(n - k) * math.factorial(k)
        out.append(num // den)

    if reverse:
        return out[::-1]
    else:
        return out


# Maximum number of iterations in Landen transformation recursion
# sequence.  10 is conservative; unit tests pass with 4, Orfanidis
# (see _arc_jac_cn [1]) suggests 5.
_ARC_JAC_SN_MAXITER = 10

#pythran export _arc_jac_sn(complex, float)
def _arc_jac_sn(w, m):
    """Inverse Jacobian elliptic sn

    Solve for z in w = sn(z, m)

    Parameters
    ----------
    w - complex scalar
        argument

    m - scalar
        modulus; in interval [0, 1]


    See [1], Eq. (56)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """

    def _complement(kx):
        # (1-k**2) ** 0.5; the expression below
        # works for small kx
        return ((1 - kx) * (1 + kx)) ** 0.5

    k = m ** 0.5

    if k > 1:
        return np.nan
    elif k == 1:
        return np.arctanh(w)

    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = _complement(k_)
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > _ARC_JAC_SN_MAXITER:
            raise ValueError('Landen transformation not converging')

    K = np.product(1 + np.array(ks[1:])) * np.pi/2

    wns = [w]

    for kn, knext in zip(ks[:-1], ks[1:]):
        wn = wns[-1]
        wnext = (2 * wn /
                 ((1 + knext) * (1 + _complement(kn * wn))))
        wns.append(wnext)

    u = 2 / np.pi * np.arcsin(wns[-1])

    z = K * u
    return z
