from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time

r = 2
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

t = np.linspace(0, 1, 1000)
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

print(time.time() - ti)
print(np.trapz(dy, x=s)/np.pi)

disp = (norms.T*dy).T
plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(y[:, 0] + disp[:, 0], y[:, 1] + disp[:, 1], label="Displaced circle after increase in Radius of 1")
plt.legend()
plt.show()