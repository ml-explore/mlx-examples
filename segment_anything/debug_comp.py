import numpy as np

x = np.load("/Users/shiyuli/Downloads/segment-anything/notebooks/tmp.npy")
y = np.load("segment_anything/tmp.npy", allow_pickle=True)
print(x.shape, y.shape)
x = x.transpose(0, 2, 3, 1)
x = x.astype(np.float32)
y = y.astype(np.float32)
# print(x)
# print()
# print(y)
print((np.abs(x - y) < 1e-1).all())
print(np.abs(x - y).max())
