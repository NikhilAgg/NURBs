from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt
import copy

n=100

points = [
    [[10, 10, 10],
    [20, 10, 20],
    [30, 10, 5],
    [40, 10, 20],
    [50, 10, 20],
    [60, 10, 10]],
    [[10, 15, 10],
    [20, 15, 20],
    [30, 15, 5],
    [40, 15, 20],
    [50, 15, 20],
    [60, 15, 10]],
    [[10, 20, 10],
    [20, 20, 20],
    [30, 20, 5],
    [40, 20, 20],
    [50, 20, 20],
    [60, 20, 10]],
    [[10, 25, 10],
    [20, 25, 20],
    [30, 25, 5],
    [40, 25, 20],
    [50, 25, 20],
    [60, 25, 10]],
    [[10, 30, 10],
    [20, 30, 20],
    [30, 30, 5],
    [40, 30, 20],
    [50, 30, 20],
    [60, 30, 10]],
]

weights = [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]]
knotsU = [0, 1/4, 1/2, 1]
multiplicitiesU = [4, 1, 1, 4]
knotsV = [0, 0.3, 0.6, 1]
multiplicitiesV = [3, 1, 1, 3]

circle = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 3,
    degreeV = 2
)
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, n)
s = np.linspace(0, 1, n)
y = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])

t = np.linspace(0, 1, n)
s = np.linspace(0, 1, n)
y = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
norms = np.array([circle.get_unit_normal(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
dp = np.array([circle.get_displacement("control point", u, v, 2, 2) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
dw = np.array([circle.get_displacement("weight", u, v, 2, 2, tie=False) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])

def func(epsilon_point, epsilon_w):
    print(f"{epsilon_w}\n-----------------------------------")
    #Changing ctrl point
    delta_p = dp[:, 0]*epsilon_point[0] + dp[:, 1]*epsilon_point[1]+ dp[:, 2]*epsilon_point[2]

    points2 = copy.deepcopy(points)
    points2[2][2] = [sum(x) for x in zip(points[2][2], epsilon_point)]
    circle = NURBsSurface(
        points2,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 2
    )
    circle.set_uniform_lc(1e-2)
    y_new_p = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
    error_p = np.array([np.dot(norms[i], dy) for i, dy in enumerate(y_new_p - y)]) - delta_p
    print(f"Max control point error: {np.max(error_p)}")

    # Changing weight
    delta_w = dw*epsilon_w

    weights2 = copy.deepcopy(weights)
    weights2[2][2] = weights2[2][2] + epsilon_w
    circle = NURBsSurface(
        points,
        weights2,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 2
    )
    circle.set_uniform_lc(1e-2)
    y_new_w = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
    error_w = np.array([np.dot(norms[i], dy) for i, dy in enumerate(y_new_w - y)]) - delta_w
    print(f"Max weight error: {np.max(error_w)}\n")

    return np.sum(np.abs(error_p)), np.sum(np.abs(error_w))


x= []
y1 = []
y2 = []
ep_step = 0.00000000001
for i in range(1, 6):
    epsilon = ep_step*i
    epsilon_point = [0.5*epsilon, epsilon, epsilon]
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