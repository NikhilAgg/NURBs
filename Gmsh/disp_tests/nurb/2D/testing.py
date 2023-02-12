from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

points = [
    [0, 0],
    [2, 1],
    [3, 1],
    [4, 0]
]

weights = [1, 1, 1, 1]
knots = [0, 1, 2]
multiplicities = [3, 1, 3]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 2, 1000)
y = np.array([circle.calculate_point(u) for u in t])
area = np.trapz(y[:, 1], x=y[:, 0])

dx = np.array([circle.get_displacement("weight", u, 1) for u in t])
s = [0]
for i in range(1, len(y)):
    s.append(s[i-1] + np.linalg.norm(y[i] - y[i-1]))
print(np.trapz(dx*0.01, x=s))

points = [
    [0, 0],
    [2, 1],
    [3, 1],
    [4, 0]
]

weights = [1, 1.01, 1, 1]
knots = [0, 1, 2]
multiplicities = [3, 1, 3]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-2)
y = np.array([circle.calculate_point(u) for u in t])
area2 = np.trapz(y[:, 1], x=y[:, 0])
print(area2-area)

# plt.title("Displacement field for a L segement with respect to control points at the mid-points")
# plt.plot(y[:, 0], y[:, 1], label = "Original")
# plt.plot(y[:, 0], dx[:, 1], label = "C for ctrl point [1, 0]")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

