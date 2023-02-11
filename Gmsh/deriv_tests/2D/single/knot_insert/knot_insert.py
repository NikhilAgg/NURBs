from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

points = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [2, -1, 0],
    [3, -1, 0],
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 2]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [4, 2, 2, 2, 4]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    3
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 100)
y = np.array([circle.calculate_point(u) for u in t])

plt.plot(y[:, 0], y[:, 1], label = "Original")

circle.knot_insertion(0.6, 1, overwrite=True)
circle.knot_insertion(0.7, 2, overwrite=True)
circle.knot_insertion(0.75, 1, overwrite=True)
t = np.linspace(0, 1, 100)
y = np.array([circle.calculate_point(u) for u in t])

plt.plot(y[:, 0], y[:, 1], "g--", label="Knot insert")
plt.legend()
plt.show()