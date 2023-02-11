from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

points = [
    [0, 0],
    [1, 0],
    [2, 0]
]

weights = [1, 1, 1]
knots = [0, 1, 2]
multiplicities = [2, 1, 2]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    1
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 2, 11)
y = np.array([circle.calculate_point(u) for u in t])
dy = np.array([circle.get_displacement("control point", u, 1) for u in t])
print(np.trapz(np.array(dy)[:, 1], x=np.array(y)[:, 0]))

plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(y[:, 0], dy[:, 1])
plt.show()