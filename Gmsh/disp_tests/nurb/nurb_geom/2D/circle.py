from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
import time
from split_circles import *

circles = [circle1, circle2, circle3, circle4]

n=25
t = np.linspace(0, 0.25, n)
# t = [0.         , 0.00970874, 0.01941748, 0.02912621, 0.03883495, 0.04854369, 0.05825243, 0.06796117, 0.0776699,  0.08737864, 0.09708738, 0.10679612,  0.11650485, 0.12621359, 0.13592233, 0.14563107, 0.15533981, 0.16504854, 0.17475728, 0.18446602, 0.19417476, 0.2038835,  0.21359223, 0.22330097, 0.23300971, 0.24271845]
# t = [0.05518335099196181, 0.10626428266100812, 0.15629766174612195, 0.20809333007820663, 0.26437677064734905, 0.3182083537885983, 0.36876187835631635, 0.4189678395467599, 0.47167589171984015, 0.5283241082801603, 0.5810321604532406, 0.6312381216436844, 0.6817916462114028, 0.735623229352652, 0.7919066699217945, 0.843702338253879, 0.8937357173389916, 0.9448166490080382]

y = np.array([circles[0].calculate_point(u) for u in t])
for circle in circles[1:]:
    p = np.array([circle.calculate_point(u) for u in t])
    y = np.r_[y, p]

norms = np.array([circle.get_unit_normal(u, flip=True) for u in t])
theta = np.arctan2(y[:, 1], y[:, 0])
s = [0]
for i in range(1, theta.shape[0]):
    s.append(s[i-1] + r*abs(abs(theta[i])-abs(theta[i-1])))

ti = time.time()

dy = np.array([np.dot(circles[0].get_displacement("control point", u, 0, flip=True), (np.array(circle1.ctrl_points[0])/r)) for u in t])
dy += np.array([np.dot(circles[0].get_displacement("control point", u, 1, flip=True), (np.array(circle1.ctrl_points[1])/r)) for u in t])
dy += np.array([np.dot(circle1.get_displacement("control point", u, 2, flip=True), (np.array(circle1.ctrl_points[2])/r)) for u in t])
C = dy
P = np.r_[C, np.zeros(((n*4)-len(C)))]
print(np.trapz(P, x=s))

for circle in circles[1:]:
    dy = np.array([np.dot(circle.get_displacement("control point", u, 0, flip=True), (np.array(circle.ctrl_points[0])/r)) for u in t])
    dy += np.array([np.dot(circle.get_displacement("control point", u, 1, flip=True), (np.array(circle.ctrl_points[1])/r)) for u in t])
    dy += np.array([np.dot(circle.get_displacement("control point", u, 2, flip=True), (np.array(circle.ctrl_points[2])/r)) for u in t])
    C = np.r_[C, dy]
    P = np.r_[C, np.zeros(((n*4)-len(C)))]
    print(np.trapz(P, x=s))

# print(time.time() - ti)
C = np.r_[C, np.zeros(((n*4)-len(C)))]
print(np.trapz(C, x=s))

# disp = (norms.T*dy).T
# plt.plot(y[:, 0], y[:, 1], label = "Original")
# # plt.plot(y[:, 0] + dy[:, 0], y[:, 1] + dy[:, 1], label="Displaced circle after increase in Radius of 1")
# plt.plot(y[:, 0] + disp[:, 0], y[:, 1] + disp[:, 1], label="Displaced circle after increase in Radius of 1")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()