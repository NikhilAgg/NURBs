from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import shapely
import time

def area_of_curve(y):
    y_ref = y[:]
    y_ref.append([60, 0])
    y_ref.append([10, 0])
    y_ref.append(y_ref[0])
    ori_shape = shapely.Polygon(y_ref)

    return ori_shape.area

def func(n, epsilon_point, epsilon_w, k):
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
    t_ref = np.linspace(0, knots[-1], 10000)
    y_ref = [circle.calculate_point(u) for u in t_ref]

    y = np.array([circle.calculate_point(u) for u in t])
    norms = np.array([circle.get_unit_normal(u, flip=flip) for u in t])
    dp = np.array([circle.get_displacement("control point", u, k, flip=flip) for u in t])
    dw = np.array([circle.get_displacement("weight", u, k, flip=flip, tie=False) for u in t])
    s = [0]
    for i in range(1, len(y)):
        s.append(s[i-1] + np.linalg.norm(y[i] - y[i-1]))

    y_ref = np.array(y_ref)
    area = np.trapz(y_ref[:, 1], x=y_ref[:, 0])

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
    y_new = np.array([circle.calculate_point(u) for u in t_ref])
    area_p = np.trapz(y_new[:, 1], x=y_new[:, 0])
    dA_p_from_dp = np.trapz(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1], x=s)
    print(f"{n}\n-----------------------------------\nControl Point - Actual vs Calculated: {area_p - area} vs {dA_p_from_dp}")
    

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
    y_new = np.array([circle.calculate_point(u) for u in t_ref])
    area_w = np.trapz(y_new[:, 1], x=y_new[:, 0])
    dA_w_from_dw = np.trapz(dw*epsilon_w, x=s)
    print(f"Weight - Actual vs Calculated: {area_w - area} vs {dA_w_from_dw}\n")

    return np.abs((area_p-area) - dA_p_from_dp), np.abs((area_w-area) - dA_w_from_dw)

epsilon_point = [1, 1]
epsilon_w = 0.1

x= []
yp = []
yw = []
for i in range(1, 5):
    xi = 10**i
    error_p, error_w = func(xi, epsilon_point, epsilon_w, 1)
    yp.append(error_p)
    yw.append(error_w)
    x.append(xi)

plt.plot(x, yp)
plt.plot(x, [0] * len(x), "k--")
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.xlabel("Number of Points along the curve")
plt.ylabel("Error")
plt.show()

plt.plot(x, yw)
plt.plot(x, [0] * len(x), "k--")
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.xlabel("Number of Points along the curve")
plt.ylabel("Error")
plt.show()