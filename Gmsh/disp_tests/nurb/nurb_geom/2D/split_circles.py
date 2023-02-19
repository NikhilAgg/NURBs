from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time
lc = 1e-2
r = 1
n=100
points = [
    [ r        ,  0.        ],
    [ r        ,  r],
    [0, r]
]

weights = [1.        , 1/2**0.5,  1]
knots = [0, 1/4]
multiplicities = [3, 3]

circle1 = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle1.set_uniform_lc(lc)

# t = np.linspace(0, 0.25, n)
# y1 = np.array([circle1.calculate_point(u) for u in t])
# plt.plot(y1[:, 0], y1[:, 1])

points = [
    [ 0.        ,  r        ],
    [ -r        ,  r],
    [-r, 0]
]
circle2 = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle2.set_uniform_lc(lc)
# t = np.linspace(0, 0.25, n)
# y2 = np.array([circle2.calculate_point(u) for u in t])
# plt.plot(y2[:, 0], y2[:, 1])
# plt.show()

points = [
    [ -r        ,  0.        ],
    [ -r        ,  -r],
    [0, -r]
]
circle3 = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle3.set_uniform_lc(lc)
# t = np.linspace(0, 0.25, n)
# y3 = np.array([circle3.calculate_point(u) for u in t])
# plt.plot(y3[:, 0], y3[:, 1])


points = [
    [ 0.        ,  -r       ],
    [ r        ,  -r],
    [r, 0]
]
circle4 = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle4.set_uniform_lc(lc)

# t = np.linspace(0, 0.25, n)
# y1 = np.array([circle1.calculate_point(u) for u in t])
# plt.plot(y1[:, 0], y1[:, 1])

# t = np.linspace(0, 0.25, n)
# y2 = np.array([circle2.calculate_point(u) for u in t])
# plt.plot(y2[:, 0], y2[:, 1])

# t = np.linspace(0, 0.25, n)
# y3 = np.array([circle3.calculate_point(u) for u in t])
# plt.plot(y3[:, 0], y3[:, 1])

# t = np.linspace(0, 0.25, n)
# y4 = np.array([circle4.calculate_point(u) for u in t])
# plt.plot(y4[:, 0], y4[:, 1])

# y = np.concatenate([y1, y2, y3, y4])
# s = [0]
# for i in range(1, y.shape[0]):
#     s.append(s[i-1] + np.linalg.norm(y[i]-y[i-1]))

# print(np.trapz(np.ones(len(s)), x=s))

# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()