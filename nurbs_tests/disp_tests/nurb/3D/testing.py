from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.integrate import dblquad

def find_ind(val, row):
    
    ind = None
    for i, x in enumerate(row):
        if val == x:
            return i
        elif val < x:
            ind = i
            break

    if ind == None:
        return -1

    frac = (val - row[ind-1])/(row[ind]-row[ind-1])
    return ind-1+frac


def from_ind(ind_frac, row):
    if ind_frac == -1:
        return 0

    ind = int(np.ceil(ind_frac))
    frac = ind_frac - ind + 1

    val = frac * (row[ind] - row[ind-1]) + row[ind-1]
    return val

points = [
    [[0, 5, 1],
    [0, 2, 1],
    [0, 0, 1]],
    [[0.5, 4, 1],
    [0.5, 2, 1],
    [0.5, 0, 1]]
]

weights = [[1, 1, 1], [1, 1, 1]]
knotsU = [0, 1, 2]
multiplicitiesU = [2, 1, 2]
knotsV = [0, 1]
multiplicitiesV = [2, 2]

circle = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 1,
    degreeV = 1
)
circle.set_uniform_lc(1e-2)

fig = plt.figure()
ax = plt.axes(projection='3d')
n=100
t = np.linspace(0, 2, n)
s = np.linspace(0, 1, n)
y = np.array([circle.calculate_point(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2])

v = []
for i in range(n):
    v.append([0])
    for j in range(1, n):
        v[i].append(v[i][j-1] + np.linalg.norm(y[i + (j-1)*n] - y[i + j*n]))

v_int = v[np.argmax(np.array(v)[:, -1])]

u = []
for i in range(n):
    u.append([0])
    for j in range(1, n):
        u[i].append(u[i][j-1] + np.linalg.norm(y[n*i + (j-1)] - y[n*i + j]))

p = []
for j, v_curr in enumerate(v_int):
    p.append([])
    for i in range(len(v)):
        ind = find_ind(v_curr, v[i])
        u_val = from_ind(ind, u[i])
        p[j].append(u_val)

p = np.array(p).T

y_mat = []
for i in range(n):
    y_mat.append(y[i*n:(i+1)*n, 2])

yp = []
for j, v_curr in enumerate(v_int):
    yp.append([])
    for i in range(len(v)):
        ind = find_ind(v_curr, v[i])
        u_val = from_ind(ind, y_mat[i])
        yp[j].append(u_val)

yp = np.array(yp).T

intg = []
for i in range(n):
    intg.append(np.trapz(yp[i], x=p[i]))

V = np.trapz(intg, x=v_int)
print(V)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()