from bspline.bspline_curve import BSplineSurface
from bspline.gmsh_utils import create_bspline_volume_mesh_file
from pathlib import Path
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

def calc(epsilon, i_der, j_der):
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
    dy = np.array([cylinder.derivative_wrt_ctrl_point(i_der, j_der, u, v)[1] for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
    der = y + dy * epsilon

    weights[j_der * cylinder.nU + i_der] += epsilon

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

    return y, der


y_tay = []
x_tay = []

ep_step = 0.0001
i_der = 3
j_der = 1

for k in range(5):
    epsilon = ep_step * k
    y, der = calc(epsilon, i_der, j_der)
    error = 0
    for i in range(len(y)):
        error += np.linalg.norm(y[i] - der[i])**2

    y_tay.append(error)
    x_tay.append(epsilon**2)


plt.plot(x_tay, y_tay)
plt.show()