import numpy as np
from numba import jit


@jit(nopython=True) 
def coo_tensor_gen(shape, density=0.02):
    nnz = int(density * shape[0] * shape[1] * shape[2])
    m = np.random.choice(shape[0], nnz)
    n = np.random.choice(shape[1], nnz)
    k = np.random.choice(shape[2], nnz)
    vals = np.random.rand(nnz)
    return np.vstack((m, n, k)).T, vals, nnz


@jit(nopython=True) 
def check_coo_tensor(coo, nnz):
    count = 0
    for i in range(nnz):
        for j in range(nnz):
            if (coo[i]==coo[j]).sum() == 3:
                count += 1
                if count > 1:
                    return "Bad"
        count = 0  
  

@jit(nopython=True) 
def mttcrp(coo_tensor, vals, nnz, shape, mode, a, b):
    temp = np.zeros(shape=(shape[mode], a.shape[1]))
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1
        
    for item in range(nnz):
        coord = coo_tensor[item]
        temp[coord[mode], :] += a[coord[mode_a], :] * b[coord[mode_b], :] * vals[item] 
    
    return temp


@jit(nopython=True) 
def cp_als3(coo_tensor,
            vals, nnz,
            shape, rank=5,
            max_iter=200,
            tol=1e-8):
    
    a = np.random.rand(shape[0], rank)
    b = np.random.rand(shape[1], rank)
    c = np.random.rand(shape[2], rank)
    err_arr = np.empty((max_iter, 1))
    
    it = 0
    err1 = 1.0
    err2 = 0.0
    while np.abs(err1 - err2) > tol:
        it += 1

        v1 = b.T @ b
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        a = mttcrp(coo_tensor, vals, nnz, shape, 0, b, c) @ v
        
        v1 = a.T @ a
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        b = mttcrp(coo_tensor, vals, nnz, shape, 1, a, c) @ v
        
        v1 = a.T @ a
        v2 = b.T @ b
        v = v1 * v2
        v = np.linalg.pinv(v)
        c = mttcrp(coo_tensor, vals, nnz, shape, 2, a, b) @ v
        
        error = sqrt_err_relative(coo_tensor, vals, nnz, shape, a, b, c)
        err_arr[it - 1] = error
        err2 = err1
        err1 = error
        if it == max_iter:
            print("iterations over")
            break
    
    return a, b, c, err_arr


@jit(nopython=True) 
def sqrt_err(coo_tensor, vals, nnz, shape, a, b, c):
    result = 0.0
    for item in range(nnz):
        coord = coo_tensor[item]
        result += np.sqrt((vals[item] - np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]))**2)        
    return result


@jit(nopython=True) 
def sqrt_err_relative(coo_tensor, vals, nnz, shape, a, b, c):
    result = sqrt_err(coo_tensor, vals, nnz, shape, a, b, c)        
    return result / np.sqrt((vals**2).sum())



