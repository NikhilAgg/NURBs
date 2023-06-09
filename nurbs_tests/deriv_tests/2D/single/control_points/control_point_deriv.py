from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

epsilon = np.array([[0, 1, 0]])
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

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
y = np.array([circle.calculate_point(u) for u in t])
dy = np.array([[circle.derivative_wrt_ctrl_point(ind_der, u)[0]] for u in t])
der = y + dy * epsilon

plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(der[:, 0], der[:, 1], "r", label="Derivative")

points = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]
], dtype=np.float32)

for i in range(3):
	points[ind_der, i] += epsilon[0][i]

circle = NURBsCurve(
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
plt.show()