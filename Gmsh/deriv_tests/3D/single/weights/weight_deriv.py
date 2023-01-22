from bspline.nurbs import NURBsSurface
from bspline.gmsh_utils import create_bspline_volume_mesh_file
from pathlib import Path
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

epsilon = 0.1
i_der = 3
j_der = 1

points = [
    [[1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]],
    [[1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [1, 0, 1]]
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
dy = np.array([cylinder.derivative_wrt_ctrl_point(i_der, j_der, u, v)[1] for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
der = y + dy * epsilon

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], 'b')
ax.scatter3D(der[:, 0], der[:, 1], der[:, 2], 'r')


weights[j_der][i_der] += epsilon

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

ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], 'g')

plt.show()