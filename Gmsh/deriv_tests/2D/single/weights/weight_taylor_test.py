from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

def calc(epsilon, ind_der):
    points = [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [-1, 1, 0],
        [-1, 0, 0],
        [-1, -1, 0],
        [0, -1, 0],
        [1, -1, 0],
        [1, 0, 0]
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

    t = np.linspace(0, 1, 100)
    y = np.array([circle.calculate_point(u) for u in t])
    dy = np.array([circle.derivative_wrt_ctrl_point(ind_der, u)[1] for u in t])
    der = y + dy * epsilon

    weights[ind_der] += epsilon

    circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(1e-2)

    t = np.linspace(0, 1, 100)
    y = np.array([circle.calculate_point(u) for u in t])

    return y, der


y_tay = []
x_tay = []

ep_step = 0.01
ind_der = 1
coord = np.array([[0, 1, 0]])

for k in range(5):
    epsilon = ep_step * k
    y, der = calc(epsilon, ind_der)
    error = 0
    for i in range(len(y)):
        error += np.linalg.norm(y[i] - der[i])

    y_tay.append(error)
    x_tay.append(epsilon**2)


plt.plot(x_tay, y_tay)
plt.show()