import numpy as np
from numba import jit


@jit(nopython=True) 
def gen_coo_tensor(shape, density=0.02):
    nnz = int(density * shape[0] * shape[1] * shape[2])
    m = np.random.choice(shape[0], nnz)
    n = np.random.choice(shape[1], nnz)
    k = np.random.choice(shape[2], nnz)
    vals = np.random.rand(nnz)
    return np.vstack((m, n, k)).T, vals


@jit(nopython=True) 
def check_coo_tensor(coo):
    count = 0
    for i in range(coo.shape[0]):
        for j in range(coo.shape[0]):
            if (coo[i]==coo[j]).sum() == 3:
                count += 1
                if count > 1:
                    return "Bad"
        count = 0  

        
@jit(nopython=True)
def gen_hilbert_tensor(shape):
    coo = []
    vals = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                coo.append((i, j, k))
                vals.append(1 / (i + j + k + 3))
    
    coo = np.array(coo)
    vals = np.array(vals)
    return coo, vals     


@jit(nopython=True) 
def sqrt_err(coo_tensor, vals, shape, a, b, c):
    result = 0.0
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        result += (vals[item] - np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]))**2        
    return np.sqrt(result)


@jit(nopython=True) 
def sqrt_err_relative(coo_tensor, vals, shape, a, b, c):
    result = sqrt_err(coo_tensor, vals, shape, a, b, c)        
    return result / np.sqrt((vals**2).sum())