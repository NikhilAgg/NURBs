from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
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
    h = x2[0]-x1[0]
    l = x3[0]-x2[0]
    t = (l**2)/(h**2)
    
    if which == "forward":
        dy_dx = ((1-t)*x1[1] + t*x2[1] - x3[1])/(h*t-l)
        dis = np.sqrt(1 + (dy_dx*h)**2)
    elif which == "backward":
        dy_dx = (x1[1] - t*x2[1] + (t-1)*x3[1])/(h*t-l)
        dis = np.sqrt(1 + (dy_dx*l)**2)

    return dis

r = 1
points = [
    [r, 0],
    [r, r],
    [0, r],
    [-r, r],
    [-r, 0],
    [-r, -r],
    [0, -r],
    [r, -r],
    [r, 0]
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
circle.set_uniform_lc(1e-3)

t = np.linspace(0, 1, 1000)
y = np.array([circle.calculate_point(u) for u in t])
area = np.trapz(y[:, 1], x=y[:, 0])

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
    s3.append(s3[i-1] + np.linalg.norm(y[i]-y[i-1]))


x= []
y = []
ep_step = 0.000001
for i in range(1, 20):
    epsilon = ep_step*i
    error = np.abs(np.abs(simpson(np.ones(len(s))*epsilon, x=s3)) - np.abs(2*epsilon*np.pi*r))
    print(error)

    y.append(error)
    x.append(epsilon**2)

plt.plot(x, y)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error")
plt.show()

