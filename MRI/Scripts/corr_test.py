import numpy as np

# if j<i: [j + i*(i+1)/2]
# else: [i + j*(j+1)/2]

class SymNDArray(np.ndarray):
    def __getitem__(self, (i, j), value):
        if j < i:
            super(SymNDArray, self).__getitem__(j + i*(i+1) / 2, value)                    
        else:
            super(SymNDArray, self).__getitem__(i + j*(j+1) / 2, value)                    

roi = np.random.rand(5,5)

roi -= roi.mean(axis=1, keepdims=True)


n = roi.shape[0]
len_ts = roi.shape[1]

tri_area =  n * (n + 1) / 2

corr = np.zeros(tri_area)

k = 0
for i in range(n):
    for j in range(i+1):
        corr[k] = ( roi[i,:].dot(roi[j,:]) / (len_ts -1) )
        k += 1      

sigma = np.zeros(n, dtype=np.float32)
k = 0
for i in range(n):
    sigma[i] = np.sqrt(corr[k])
    k += (i+2)


k = 0
for i in range(n):
    for j in range(i+1):
        corr[k] /= (sigma[i] * sigma[j])    
        k += 1


corr = corr.astype(SymNDArray)
corr[2]