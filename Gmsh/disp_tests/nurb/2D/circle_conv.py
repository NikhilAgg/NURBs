from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time

def func(n, r):
    points = [
        [r, 0],
        [r, r],
        [0, r],
        [-r, r],
        [-r, 0],
        [-r, -r],
        [0, -r],
        [r, -r],
        [r, 0]
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

    t = np.linspace(0, 1, n)
    y = np.array([circle.calculate_point(u) for u in t])
    norms = np.array([circle.get_unit_normal(u, flip=True) for u in t])
    theta = np.arctan2(y[:, 1], y[:, 0])
    s = [0]
    for i in range(1, theta.shape[0]):
        s.append(s[i-1] + r*abs(abs(theta[i])-abs(theta[i-1])))

    ti = time.time()
    dy = np.array([np.linalg.norm(circle.get_displacement("control point", u, 0, flip=True)) for u in t])
    for i in range(1, 8):
        p = np.array([np.linalg.norm(circle.get_displacement("control point", u, i, flip=True)) for u in t])
        dy += p

    return np.trapz(dy, x=s)


x= []
y = []
r = 2
for i in range(1, 6):
    xi = 10**i
    y.append(func(xi, r))
    x.append(xi)

plt.plot(x, y)
plt.plot(x, [2*np.pi*r]*len(x), label="$2\pi r$")
plt.xscale("log", base=10)
plt.xlabel("Number of Points along the curve")
plt.ylabel("Integral of Displacement field")
plt.legend()
plt.show()