from bspline.bspline_curve import BSplineSurface
from bspline.gmsh_utils import create_bspline_volume_mesh_file
from pathlib import Path
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

points = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [1, 0, 1]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1]
multiplicitiesU = [3, 2, 2, 2, 3]
multiplicitiesV = [2, 2]

cylinder = BSplineSurface(
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

dt = np.linspace(0, 1, 10)
ds = np.linspace(0, 1, 10)
dy = []
arrow_y = []
for u, v in np.array(np.meshgrid(dt, ds)).T.reshape(-1,2):
    ders = cylinder.derivative_wrt_uv(u, v, 1)
    arrow_y.append(ders[0][0])
    dy.append(np.cross(ders[1][0], ders[0, 1]))

arrow_y = np.array(arrow_y)
dy = np.array(dy)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], 'b')

arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
for i in range(len(dy)):
    a = Arrow3D([arrow_y[i, 0], arrow_y[i, 0] + dy[i,0]/10], [arrow_y[i, 1], arrow_y[i, 1] + dy[i,1]/10], [arrow_y[i, 2], arrow_y[i, 2] + dy[i,2]/10], **arrow_prop_dict)
    ax.add_artist(a)

plt.show()