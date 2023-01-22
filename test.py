import numpy as np

# x = np.array([[[0], [1]], [[2], [3]]])
x = np.array([[0], [1], [2], [3]])
# y = np.array([[0, 3], [4, 5]])
y = np.array([0, 3, 4, 5])

x = x.reshape(-1, 1)
for i, j in zip(x, y.flatten()):
    print(i, j)