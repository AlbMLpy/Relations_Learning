import numpy as np
from numba import jit

from general_functions import sqrt_err_relative


@jit(nopython=True)
def generate_system(coo_tensor, vals, shape, mode, a, b, l2, step):
    mtx = np.zeros((a.shape[1], a.shape[1]))
    right = np.zeros((a.shape[1]))
    mask = coo_tensor[:, mode] == step
    coo_step = coo_tensor[mask]
    vals_step = vals[mask]
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1

#    temp_right = 0.0
#    for i in range(a.shape[1]):
#        a_i = a[:, i]
#        b_i = b[:, i]
#        for j in range(a.shape[1]):
#            a_j = a[:, j]
#            b_j = b[:, j]
#            temp_mtx = 0.0
#            for item in range(coo_step.shape[0]):
#                coord = coo_step[item]
#                coo_a = coord[mode_a]
#                coo_b = coord[mode_b]
                
#                temp_mtx += (a_i[coo_a] 
#                             * b_i[coo_b] 
#                             * a_j[coo_a] 
#                             * b_j[coo_b])
#                
 #               if i == 0:
#                    temp_right += a_j[coo_a] * b_j[coo_b] * vals_step[item]
                    
#            if i == 0:
#                right[j] = temp_right
#                temp_right = 0
                
#            mtx[i, j] = temp_mtx
#            if i == j:
#                mtx[i, j] += l2        
    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
            for item in range(coo_step.shape[0]):
                coord = coo_step[item]
                mtx[i, j] += (a[coord[mode_a], i] 
                              * b[coord[mode_b], i] 
                              * a[coord[mode_a], j] 
                              * b[coord[mode_b], j])
                if i == 0:
                    right[j] += a[coord[mode_a], j] * b[coord[mode_b], j] * vals_step[item]
            
            if i == j:
                mtx[i, j] += l2
    
    return mtx, right


@jit(nopython=True) 
def wrcp_als3(coo_tensor,
              vals,
              shape,
              rank=5,
              l2=0.5,
              max_iter=50,
              tol=1e-8,
              seed=13,
              show_iter=False,
              it_over=True,
              ):
    
    random_state = np.random.seed(seed)#np.random if seed is None else np.random.RandomState(seed)
    
    a = np.random.normal(0.0, 0.1, size=(shape[0], rank))
    b = np.random.normal(0.0, 0.1, size=(shape[1], rank))
    c = np.random.normal(0.0, 0.1, size=(shape[2], rank))
    err_arr = np.empty((max_iter, 1))  
    
    it = 0
    err1 = 1.0
    err2 = 0.0
    while np.abs(err1 - err2) > tol:
        it += 1
        
        for i in range(shape[0]):
            A, right = generate_system(
                coo_tensor, vals, shape,
                0, b, c, l2, i,
            )
            
            #a[i, :] = np.linalg.pinv(A) @ right
            a[i, :] = np.linalg.solve(A, right)
            
        for j in range(shape[1]):
            A, right = generate_system(
                coo_tensor, vals, shape,
                1, a, c, l2, j,
            )
            
            #b[j :] = np.linalg.pinv(A) @ right
            b[j :] = np.linalg.solve(A, right)
            
        for k in range(shape[2]):   
            A, right = generate_system(
                coo_tensor, vals, shape,
                2, a, b, l2, k,
            )
            
            #c[k, :] = np.linalg.pinv(A) @ right
            c[k, :] = np.linalg.solve(A, right)
    
        error = sqrt_err_relative(coo_tensor, vals, shape, a, b, c)
        err_arr[it - 1] = error
        err2 = err1
        err1 = error
        if show_iter:
            print("Iter: ", it, "; Error: ", error)
            
        if it == max_iter:
            if it_over:
                print("iterations over")
            break
    
    return a, b, c, err_arr, it