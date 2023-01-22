from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt

def calc(u, v, order):
    points = [
        [[1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [-1, 1, 0],
        [-1, 0, 0],
        [-1, -1, 0],
        [0, -1, 0],
        [1, -1, 0],
        [1, 0, 0]],
        [[1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [-1, 1, 1],
        [-1, 0, 1],
        [-1, -1, 1],
        [0, -1, 1],
        [1, -1, 1],
        [1, 0, 1]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
    knotsU = [0, 1/4, 1/2, 3/4, 1]
    knotsV = [0, 1]
    multiplicitiesU = [3, 2, 2, 2, 3]
    multiplicitiesV = [2, 2]

    cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    cylinder.set_uniform_lc(1e-2)

    y = cylinder.calculate_point(u, v)
    dy = cylinder.derivative_wrt_uv(u, v, order)

    return y, dy

def get_der(y, dy, epsilon, wrt):
    der = y.copy()
    for i in range(1, order+1):
        if wrt == 0:
            temp = dy[i][0]*(epsilon**i) / np.math.factorial(i)
        elif wrt == 1:
            temp = dy[0][i]*(epsilon**i) / np.math.factorial(i)
        der += temp

    return der

y_tay = []
x_tay = []

ep_step = 0.01
u = 0.7
v = 0.5
wrt = 0
order = 2
coord = np.array([[0, 1, 0]])

y_orig, dy = calc(u, v, order)

for k in range(10):
    epsilon = ep_step * k
    if wrt == 0:
        y = calc(u+epsilon, v, order)[0]
    elif wrt == 1:
        y = calc(u, v+epsilon, order)[0]
    der = get_der(y_orig, dy, epsilon, wrt)
    error = 0
    for i in range(len(y)):
        error += np.linalg.norm(y[i] - der[i])**2

    y_tay.append(error)
    x_tay.append(epsilon**(order+1))


plt.plot(x_tay, y_tay)
plt.show()