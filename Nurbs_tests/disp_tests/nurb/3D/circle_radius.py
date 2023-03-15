from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt
import time

r = 5
points = [
    [[r, 0, 0],
    [r, r, 0],
    [0, r, 0],
    [-r, r, 0],
    [-r, 0, 0],
    [-r, -r, 0],
    [0, -r, 0],
    [r, -r, 0],
    [r, 0, 0]],
    [[r, 0, 1],
    [r, r, 1],
    [0, r, 1],
    [-r, r, 1],
    [-r, 0, 1],
    [-r, -r, 1],
    [0, -r, 1],
    [r, -r, 1],
    [r, 0, 1]]
]

weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1]
multiplicitiesU = [3, 2, 2, 2, 3]
multiplicitiesV = [2, 2]

cylinder = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 2,
    degreeV = 1
)
cylinder.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
s = np.linspace(0, 1, 100)
y = np.array([cylinder.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
norms = np.array([cylinder.get_unit_normal(u, v, flip_norm=True) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
# theta = np.arctan2(y[:, 1], y[:, 0])

ti = time.time()
dy = np.zeros(len(y))
for j in range(2):
    for i in range(8):
        p = np.array([np.sum(cylinder.get_displacement("control point", u, v, j, i, flip_norm=True)[0:2]*((np.array(points[j][i])/r)[0:2])) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
        dy += p

print(time.time()-ti)

disp = (norms.T*dy).T
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2])
ax.scatter3D(y[:, 0] + disp[:, 0], y[:, 1] + disp[:, 1], y[:, 2] + disp[:, 2])
ax.scatter3D(np.zeros(10), np.linspace(0, r+1, 10), np.zeros(10))
plt.show()