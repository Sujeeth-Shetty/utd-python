
import sys
sys.path.append("NumPy_path")

import numpy as np

n = 1000
x = np.random.normal(size=(n,2))
y = np.random.normal(size=(n,1))
ones = np.ones((n,1))
x = np.hstack([ones,x])
b = np.array([1,2,0]).reshape(-1,1)
y += x@b

xx = np.matmul(x.T,x)
xx = x.T@x
xy = x.T@y
b = np.linalg.inv(xx)@xy
b = np.linalg.solve(xx,xy) 
# np.linalg.solve(A,b) = A^-1 b
res = y-x@b
rsq = 1-res.var()/y.var()
vb = res.var()*np.linalg.inv(xx)
se = np.sqrt(np.diagonal(vb)).reshape(-1,1)
tstat = b/se

XtX_inv= np.linalg.inv(xx)
A=XtX_inv.dot(np.transpose(x))
B=x.dot(XtX_inv)
res_hstack=np.hstack(res)
res_sq=np.diag(res_hstack**2)
std_dev_list=np.repeat(res.std(),n-1)
np.fill_diagonal(res_sq[1:], std_dev_list)
np.fill_diagonal(res_sq[:,1:], -std_dev_list)
VHCE=A.dot(res_sq).dot(B)
print(VHCE)