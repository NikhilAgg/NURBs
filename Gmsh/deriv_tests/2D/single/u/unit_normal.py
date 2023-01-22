from bspline.nurbs import NURBsCurve, NURBs2DGeometry
from bspline.gmsh_utils import create_bspline_curve_mesh_file
import numpy as np
import matplotlib.pyplot as plt
import os

epsilon = 0.1
ind_der = 3

points = [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [3, 2, 2, 2, 3]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)

# points = [
#     [0, 0],
#     [0, 1]
# ]

# weights = [1, 1]
# knots = [0, 1]
# multiplicities = [2, 2]

# circle = BSplineCurve(
#     points,
#     weights,
#     knots,
#     multiplicities,
#     1
# )
# circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
dt = np.linspace(0, 1, 30)
y = np.array([circle.calculate_point(u) for u in t])
arrow_y = np.array([circle.calculate_point(u) for u in dt])
dy = np.array([circle.derivative_wrt_u(u, 1)[1] for u in dt])

plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.quiver(arrow_y[:, 0], arrow_y[:, 1], -dy[:, 1], dy[:, 0])
plt.show()