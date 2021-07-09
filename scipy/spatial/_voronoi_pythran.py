import numpy as np


# array-filling placeholder that can never occur
ARRAY_FILLER = -2

#pythran export remaining_filter(int[:], int)
def remaining_filter(remaining, current_simplex):
    for i in range(remaining.shape[0]):
        if remaining[i] == current_simplex:
            remaining[i] = ARRAY_FILLER

#pythran export sort_vertices_of_regions(int[:,:], int list list)
#pythran export sort_vertices_of_regions(int32[:,:], int64 list list)
def sort_vertices_of_regions(simplices, regions):
    sorted_vertices = np.empty(max([len(region) for region
                               in regions]),
                               dtype=np.intp)

    for n in range(len(regions)):
        remaining_count = 0
        remaining = np.asarray(regions[n][:])
        remaining_size = remaining.shape[0]
        sorted_vertices[:] = ARRAY_FILLER
        current_simplex = remaining[0]
        for i in range(3):
            k = simplices[current_simplex, i]
            if k != n:
                current_vertex = k
                break
        sorted_vertices[remaining_count] = current_simplex
        remaining_count += 1
        remaining_filter(remaining, current_simplex)
        while remaining_size > remaining_count:
            cs_identified = 0
            for i in range(remaining_size):
                if remaining[i] == ARRAY_FILLER:
                    continue
                s = remaining[i]
                for j in range(3):
                    if current_vertex == simplices[s, j]:
                        current_simplex = remaining[i]
                        cs_identified += 1
                        break
                if cs_identified > 0:
                    break
            for i in range(3):
                s = simplices[current_simplex, i]
                if s != n and s != current_vertex:
                    current_vertex = s
                    break
            sorted_vertices[remaining_count] = current_simplex
            remaining_count += 1
            remaining_filter(remaining, current_simplex)
        regions_arr = np.asarray(sorted_vertices)
        regions[n] = list(regions_arr[regions_arr > ARRAY_FILLER])
    return regions