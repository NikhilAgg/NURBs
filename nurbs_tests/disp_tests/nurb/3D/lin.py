from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.integrate import dblquad

points = [
    [[0, 4, 0],
    [0, 2, 0],
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0]],
    [[0, 4, 0.5],
    [0, 2, 0.5],
    [0, 0, 0.5],
    [1, 0, 0.5],
    [2, 0, 0.5]],
    [[0, 4, 1],
    [0, 2, 1],
    [0, 0, 1],
    [1, 0, 1],
    [2, 0, 1]]
]

weights = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
knotsU = [0, 1, 2, 3, 4]
multiplicitiesU = [2, 1, 1, 1, 2]
knotsV = [0, 1, 2]
multiplicitiesV = [2, 1, 2]

circle = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 1,
    degreeV = 1
)
circle.set_uniform_lc(1e-2)

fig = plt.figure()
ax = plt.axes(projection='3d')

t = np.linspace(0, 4, 10)
s = np.linspace(0, 2, 10)
y = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2])

t = np.linspace(0, 4, 20)
s = np.linspace(0, 2, 20)
y = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
dy = np.array([circle.get_displacement("control point", u, v, 1, 1) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
dx = np.array([circle.get_displacement("control point", u, v, 1, 3) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
ax.scatter3D(dy[:, 0], y[:, 1], y[:, 2])
ax.scatter3D(y[:, 0], dx[:, 1], y[:, 2])

plt.show()