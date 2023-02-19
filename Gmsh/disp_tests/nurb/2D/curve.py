from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time

epsilon_point = [1, 1]
epsilon_w = 0.3
r=5
points = [
    [r, 0],
    [r+1, r+1],
    [0, r-2],
    [-r-1, r+1],
    [-r+1, 0],
    [-r-1, -r-1],
    [0, -r+1],
    [r+1, -r-2],
    [r-2, -0.1],
    [r, 0]
]

weights = [1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 1]
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

flip = False
t = np.linspace(0, 1, 1000)
y = np.array([circle.calculate_point(u) for u in t])
norms = np.array([circle.get_unit_normal(u, flip=flip) for u in t])
dp = np.array([circle.get_displacement("control point", u, 2, flip=flip) for u in t])
dw = np.array([circle.get_displacement("weight", u, 2, flip=flip) for u in t])
ds = [0]
for i in range(1, len(y)):
    ds.append(ds[i-1] + np.linalg.norm(y[i] - y[i-1]))

area = np.trapz(y[:, 1], x=y[:, 0])



#Changing ctrl point
points2 = points[:][:]
points2[2] = [sum(x) for x in zip(points[2], epsilon_point)]
circle = NURBsCurve(
    points2,
    weights,
    knots,
    multiplicities,
    3
)
circle.set_uniform_lc(1e-2)
y_new_p = np.array([circle.calculate_point(u) for u in t])
area_new = np.trapz(y_new_p[:, 1], x=y_new_p[:, 0])
dp_ep = np.array([epsilon_point[0] * x + epsilon_point[1] * y for x, y in dp])
area_calc_p = np.trapz(dp_ep, x=ds)

#Changing ctrl point
weights2 = weights[:]
weights2[2] += epsilon_w
circle = NURBsCurve(
    points,
    weights2,
    knots,
    multiplicities,
    3
)
circle.set_uniform_lc(1e-2)
y_new_w = np.array([circle.calculate_point(u) for u in t])
area_new = np.trapz(y_new_w[:, 1], x=y_new_w[:, 0])
area_calc_w = np.trapz(dw*epsilon_w, x=ds)


#PLotting control point change
disp = (norms.T*dw).T * epsilon_w
disp2 = (norms.T*(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1])).T
plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(y_new_p[:, 0], y_new_p[:, 1], label = "Actual")
plt.plot(y[:, 0] + disp2[:, 0], y[:, 1] + disp2[:, 1], label = "Displaced curve due to control point")

plt.plot([y[0, 0], y[0, 0]], [0, y[0, 1]], "k--")
plt.plot([y[-1, 0], y[-1, 0]], [0, y[-1, 1]], "k--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(ymin=0)
plt.show()

#Plotting weight change
disp = (norms.T*dw).T * epsilon_w
disp2 = (norms.T*(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1])).T
plt.plot(y[:, 0], y[:, 1], label = "Original")
plt.plot(y_new_w[:, 0], y_new_w[:, 1], label = "Actual")
plt.plot(y[:, 0] + disp[:, 0], y[:, 1] + disp[:, 1], label="Displaced curve due to weight")

plt.plot([y[0, 0], y[0, 0]], [0, y[0, 1]], "k--")
plt.plot([y[-1, 0], y[-1, 0]], [0, y[-1, 1]], "k--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(ymin=0)
plt.show()