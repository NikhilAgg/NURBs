from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time
k = 1
n = 10000
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

flip = False
t = np.linspace(0, knots[-1], n)

y = np.array([circle.calculate_point(u) for u in t])
norms = np.array([circle.get_unit_normal(u, flip=flip) for u in t])
dp = np.array([circle.get_displacement("control point", u, k, flip=flip) for u in t])
dw = np.array([circle.get_displacement("weight", u, k, flip=flip) for u in t])

s = [0]
for i in range(1, len(y)):
    s.append(s[i-1] + np.linalg.norm(y[i] - y[i-1]))

area = np.trapz(y[:, 1], x=y[:, 0])

def func(epsilon_point, epsilon_w):
    #Changing ctrl point
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

    y_new = np.array([circle.calculate_point(u) for u in t])
    area_p = np.trapz(y_new[:, 1], x=y_new[:, 0])
    dA_p_from_dp = np.trapz(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1], x=s)
    print(f"{epsilon_w}\n-----------------------------------\nControl Point - Actual vs Calculated: {(area_p - area) - dA_p_from_dp}")

    #Changing weight
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
    y_new = np.array([circle.calculate_point(u) for u in t])
    area_w = np.trapz(y_new[:, 1], x=y_new[:, 0])
    dA_w_from_dw = np.trapz(dw*epsilon_w, x=s)
    print(f"Weight - Actual vs Calculated: {area_w - area} vs {dA_w_from_dw}\n")

    return area_p-area, area_w-area

x= [0]
y1 = [0]
y2 = [0]
ep_step = 0.01
for i in range(1, 5):
    epsilon = ep_step*i
    epsilon_point = [0, epsilon]
    dA_p, dA_w = func(epsilon_point, epsilon)

    dA_p_from_dp = np.trapz(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1], x=s)
    dA_w_from_dw = np.trapz(dw*epsilon, x=s)

    y1.append(np.abs(dA_p - dA_p_from_dp))
    y2.append(np.abs(dA_w - dA_w_from_dw))
    x.append(epsilon**2)

plt.plot(x, y1)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error in calculated area")
plt.show()

plt.plot(x, y2)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error in calculated area")
plt.show()