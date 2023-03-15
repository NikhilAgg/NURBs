from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import simpson, quad

k = 1
n = 10001
points = [
    [10, 10],
    [20, 20],
    [30, 5],
    [40, 20],
    [50, 20],
    [60, 10]
]

weights = [1, 2, 3, 4, 5, 6]
knots = [0, 1/4, 1/2, 1]
multiplicities = [4, 1, 1, 4]
degree = 3

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    degree
)
circle.set_uniform_lc(1e-2)
t = np.linspace(0, knots[-1], 1000000)
y = np.array([circle.calculate_point(u) for u in t])

flip = False
t = np.linspace(0, knots[-1], n)

y = np.array([circle.calculate_point(u) for u in t])
norms = np.array([circle.get_unit_normal(u, flip=flip) for u in t])
dp = np.array([circle.get_displacement("control point", u, k, flip=flip) for u in t])
dw = np.array([circle.get_displacement("weight", u, k, flip=flip, tie=False) for u in t])

# s = [0]
# s.append(get_dis(y[0], y[1], y[2], which=0))
# for i in range(2, len(y)):
#     s.append(s[i-1] + get_dis(y[i-2], y[i-1], y[i]))



area = simpson(y[:, 1], x=y[:, 0])

def func(epsilon_point, epsilon_w):
    print(f"{epsilon_w}\n-----------------------------------")
    
    #Changing ctrl point
    delta_p = dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1]

    points2 = points[:][:]
    points2[k] = [sum(x) for x in zip(points[k], epsilon_point)]

    circle = NURBsCurve(
        points2,
        weights,
        knots,
        multiplicities,
        degree
    )
    circle.set_uniform_lc(1e-2)

    y_new_p = np.array([circle.calculate_point(u) for u in t])
    error_p = np.array([np.dot(norms[i], dy) for i, dy in enumerate(y_new_p - y)]) - delta_p
    print(f"Max control point error: {np.max(error_p)}")

    #Changing weight
    delta_w = dw*epsilon_w

    weights2 = weights[:]
    weights2[k] += epsilon_w
    circle = NURBsCurve(
        points,
        weights2,
        knots,
        multiplicities,
        degree
    )
    circle.set_uniform_lc(1e-2)
    y_new_w = np.array([circle.calculate_point(u) for u in t])
    error_w = np.array([np.dot(norms[i], dy) for i, dy in enumerate(y_new_w - y)]) - delta_w
    print(f"Max weight error: {np.max(error_w)}\n")

    return np.sum(np.abs(error_p)), np.sum(np.abs(error_w))


x= []
y1 = []
y2 = []
ep_step = 0.01
for i in range(1, 5):
    epsilon = ep_step*i
    epsilon_point = [0, epsilon]
    error_p, error_w = func(epsilon_point, epsilon)

    y1.append(error_p)
    y2.append(error_w)
    x.append(epsilon**2)

plt.plot(x, y1)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Total error in displacement")
plt.show()

plt.plot(x, y2)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Total error in displacement")
plt.show()