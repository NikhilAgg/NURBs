from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time

r = 1
n=2000
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
# t = [0.05518335099196181, 0.10626428266100812, 0.15629766174612195, 0.20809333007820663, 0.26437677064734905, 0.3182083537885983, 0.36876187835631635, 0.4189678395467599, 0.47167589171984015, 0.5283241082801603, 0.5810321604532406, 0.6312381216436844, 0.6817916462114028, 0.735623229352652, 0.7919066699217945, 0.843702338253879, 0.8937357173389916, 0.9448166490080382]

y = np.array([circle.calculate_point(u) for u in t])
norms = np.array([circle.get_unit_normal(u, flip=True) for u in t])
theta = np.arctan2(y[:, 1], y[:, 0])
s = [0]
for i in range(1, theta.shape[0]):
    s.append(s[i-1] + r*abs(abs(theta[i])-abs(theta[i-1])))

ti = time.time()

dy = np.array([np.dot(circle.get_displacement("control point", u, 0, flip=True), (np.array(points[0])/r)) for u in t])
# print(np.trapz(dy, x=s))
for i in range(1, 8):
    p = np.array([np.sum(circle.get_displacement("control point", u, i, flip=True)*(np.array(points[i])/r)) for u in t])
    print(np.trapz(p, x=s))
    dy += p

# print(time.time() - ti)
print(np.trapz(dy, x=s))

# disp = (norms.T*dy).T
# plt.plot(y[:, 0], y[:, 1], label = "Original")
# # plt.plot(y[:, 0] + dy[:, 0], y[:, 1] + dy[:, 1], label="Displaced circle after increase in Radius of 1")
# plt.plot(y[:, 0] + disp[:, 0], y[:, 1] + disp[:, 1], label="Displaced circle after increase in Radius of 1")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()