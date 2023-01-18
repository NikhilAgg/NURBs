from bspline.bspline_curve import BSplineCurve, BSpline2DGeometry
from bspline.gmsh_utils import create_bspline_curve_mesh_file
import numpy as np
import matplotlib.pyplot as plt
import os

epsilon = 0.2
ind_der = 3

points = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [3, 2, 2, 2, 3]

circle = BSplineCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
y = np.array([circle.calculate_point(u) for u in t])
dy = np.array([circle.derivative_wrt_knot(ind_der, u) for u in t])
der = y + dy * epsilon

plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(der[:, 0], der[:, 1], "r", label="Derivative")


if multiplicities[ind_der] > 1:
    multiplicities[ind_der] -= 1
    multiplicities.insert(ind_der, 1)
    knots.insert(ind_der + 1, knots[ind_der] + epsilon)
else:
    knots[ind_der] += epsilon

circle = BSplineCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
y = np.array([circle.calculate_point(u) for u in t])

plt.plot(y[:, 0], y[:, 1], "g--", label="Actual")
plt.legend()


knot_y = np.array([circle.calculate_point(u) for u in knots])
plt.scatter(knot_y[:, 0], knot_y[:, 1])

plt.show()