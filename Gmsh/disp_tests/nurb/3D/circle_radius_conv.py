from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt
import time

def func(n,r, l):
    points = [
        [[r, 0, 0],
        [r, r, 0],
        [0, r, 0],
        [-r, r, 0],
        [-r, 0, 0],
        [-r, -r, 0],
        [0, -r, 0],
        [r, -r, 0],
        [r, 0, 0]],
        [[r, 0, l],
        [r, r, l],
        [0, r, l],
        [-r, r, l],
        [-r, 0, l],
        [-r, -r, l],
        [0, -r, l],
        [r, -r, l],
        [r, 0, l]]
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

    t = np.linspace(0, 1, n)
    s = np.linspace(0, 1, n)
    y = np.array([cylinder.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
    norms = np.array([cylinder.get_unit_normal(u, v, flip=True) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
    theta = np.arctan2(y[0::n, 1], y[0::n, 0])
    sl = [0]
    for i in range(1, theta.shape[0]):
        sl.append(sl[i-1] + r*abs(abs(theta[i])-abs(theta[i-1])))

    ti = time.time()
    dy = np.zeros(len(y))
    for j in range(2):
        for i in range(8):
            p = np.array([np.sum(cylinder.get_displacement("control point", u, v, j, i, flip=False)[0:2]*((np.array(points[j][i])/r)[0:2])) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
            dy += p

    intg = []
    for i in range(n):
        intg.append(np.trapz(dy[i*n:(i+1)*n], x = y[i*n:(i+1)*n, 2]))

    dV = np.trapz(intg, x=sl)
    return dV

x= []
y = []
y_cache = [9.991080713140022, 1.7521443777203558, 0.8621414607654287, 0.5717123965035285, 0.4276476737890391]
r = 5
l = 3
xs = [10, 50, 100, 150, 200]
# for i in range(0, 5):
#     print(i)
#     xi = xs[i]
#     y.append(np.abs(func(xi, r, l) - 2*np.pi*r*l))
#     x.append(xi)
# print(x)
# print(y)
# plt.plot(x, y)
plt.plot(xs, y_cache)
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.xlabel("Number of Points along the curve")
plt.ylabel("Log Error")
plt.legend()
plt.show()