from bspline.bspline_curve import BSplineCurve
import numpy as np
import matplotlib.pyplot as plt

def calc(u, order):
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

    circle = BSplineCurve(
        points,
        weights,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(1e-2)

    y = circle.calculate_point(u)
    dy = circle.derivative_wrt_u(u, order)

    return y, dy

def get_der(y, dy, epsilon):
    der = y.copy()
    for i in range(1, order+1):
        temp = dy[i]*(epsilon**i) / np.math.factorial(i)
        der += temp

    return der

y_tay = []
x_tay = []

ep_step = 0.01
u = 0.7
order = 2
coord = np.array([[0, 1, 0]])

y_orig, dy = calc(u, order)

for k in range(10):
    epsilon = ep_step * k
    y = calc(u+epsilon, order)[0]
    der = get_der(y_orig, dy, epsilon)
    error = 0
    for i in range(len(y)):
        error += np.linalg.norm(y[i] - der[i])**2

    y_tay.append(error)
    x_tay.append(epsilon**(order+1))


plt.plot(x_tay, y_tay)
plt.show()