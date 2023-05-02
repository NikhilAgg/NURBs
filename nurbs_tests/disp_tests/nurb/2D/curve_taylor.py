from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import simpson, quad

def get_dis2(x1, x2, x3, which="backward"):
    coeffs = np.polyfit([x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]], deg=2)
    def func(x):
        return (4*coeffs[0]**2*x**2 + 4*coeffs[0]*coeffs[1]*x + coeffs[1]**2 + 1)**0.5

    if which == "forward":
        return quad(func, x1[0], x2[0], epsabs=1e-19)[0]
    else:
        return quad(func, x2[0], x3[0], epsabs=1e-19)[0]

def get_dis(x1, x2, x3, which="backward"):
    dx = np.mean([x2[0]-x1[0], x3[0]-x2[0]])
    
    if which == "forward":
        dy_dx = (-3*x1[1] + 4*x2[1] - x3[1])/(2*dx)
    elif which == "backward":
        dy_dx = (x1[1] - 4*x2[1] + 3*x3[1])/(2*dx)

    dis = np.sqrt(1 + dy_dx**2)*(dx)
    return dis
    

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

flip = False
t = np.linspace(0, knots[-1], n)

y = np.array([circle.calculate_point(u) for u in t])
norms = np.array([circle.get_unit_normal(u, flip=flip) for u in t])
dp = np.array([circle.get_displacement("control point", u, k, flip=flip) for u in t])
dw = np.array([circle.get_displacement("weight", u, k, flip=flip, tie=False) for u in t])

s = [0]
s.append(get_dis(y[1], y[2], y[3], which="forward"))
for i in range(2, len(y)):
    s.append(s[i-1] + get_dis(y[i-2], y[i-1], y[i], which="backward"))

s2 = [0]
s2.append(get_dis2(y[1], y[2], y[3], which="forward"))
for i in range(2, len(y)):
    s2.append(s2[i-1] + get_dis2(y[i-2], y[i-1], y[i], which="backward"))

s3 = [0]
s3.append(np.linalg.norm(y[1]-y[0]))
for i in range(2, len(y)):
    s3.append(s[i-1] + np.linalg.norm(y[i]-y[i-1]))

area = simpson(y[:, 1], x=y[:, 0])

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
    area_p = simpson(y_new[:, 1], x=y_new[:, 0])
    dA_p_from_dp = simpson(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1], x=s)
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
    area_w = simpson(y_new[:, 1], x=y_new[:, 0])
    dA_w_from_dw = simpson(dw*epsilon_w, x=s)
    print(f"Weight - Actual vs Calculated: {area_w - area} vs {dA_w_from_dw}\n")

    return area_p-area, area_w-area

x= [0]
y1 = [0]
y2 = [0]
ep_step = 0.000001
for i in range(1, 5):
    epsilon = ep_step*i
    epsilon_point = [0, epsilon]
    dA_p, dA_w = func(epsilon_point, epsilon)

    dA_p_from_dp = simpson(dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1], x=s)
    dA_w_from_dw = simpson(dw*epsilon, x=s)

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