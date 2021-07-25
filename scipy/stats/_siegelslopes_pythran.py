import numpy as np


# pythran export siegelslopes(int[:] or float[:],
#                             int[:] or float[:] or None,
#                             str)
def siegelslopes(y, x, method):
    if method not in ['hierarchical', 'separate']:
        raise ValueError("method can only be 'hierarchical' or 'separate'")
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" %
                             (len(y), len(x)))

    deltax = np.expand_dims(x, 1) - x
    deltay = np.expand_dims(y, 1) - y
    slopes, intercepts = [], []

    for j in range(len(x)):
        id_nonzero, = np.nonzero(deltax[j, :])
        slopes_j = np.empty(len(id_nonzero))
        for i in range(len(id_nonzero)):
            slopes_j[i] = deltay[j, id_nonzero[i]] / deltax[j, id_nonzero[i]]
        slopes_j = deltay[j, id_nonzero] / deltax[j, id_nonzero]
        medslope_j = np.median(slopes_j)
        slopes.append(medslope_j)
        if method == 'separate':
            z = y*x[j] - y[j]*x
            intercept_j = np.empty(len(id_nonzero))
            for i in range(len(id_nonzero)):
                intercept_j[i] = z[id_nonzero[i]] / deltax[j, id_nonzero[i]]
            medintercept_j = np.median(intercept_j)
            intercepts.append(medintercept_j)

    medslope = np.median(np.asarray(slopes))
    if method == "separate":
        medinter = np.median(np.asarray(intercepts))
    else:
        medinter = np.median(y - medslope*x)

    return medslope, medinter
