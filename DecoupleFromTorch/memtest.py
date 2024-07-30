import numpy as np
import cupy as cp

N = 1000
K = 1000
M = 1000

A = cp.random.rand(N, K).astype(cp.float16)
B = cp.random.rand(K, M).astype(cp.float16)

print(A.__cuda_array_interface__['data'][0], A.data.ptr)
print(B.__cuda_array_interface__['data'][0], B.data.ptr)

a = torch.randn((N, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, M), device='cuda', dtype=torch.float16)

print(a.data_ptr())
print(b.data_ptr())

import matrixmult as tmm

c_triton = tmm.matmul(A, B)
c_torch = torch.matmul(A, B)

print(cp.allclose(c_triton, c_torch))